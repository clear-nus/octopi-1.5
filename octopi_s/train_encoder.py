import os 
import torch.nn as nn 
import torch
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import json
import numpy as np
from utils.dataset import *
from utils.llm import *
from utils.encoder import *
from utils.physiclear_constants import get_categorical_labels
from utils.dataset import get_dataset_sensor_type
import torch.nn.functional as F
from transformers import AutoTokenizer
import random
import yaml
from datetime import datetime
from transformers import CLIPImageProcessor, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from accelerate import infer_auto_device_map, init_empty_weights
import wandb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from sklearn import metrics



def setup(rank, world_size):
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_fn(train_fn, world_size, configs, exp_id, g, device, train_tactile_cont_dataset, train_text_cont_dataset, exp_name):
    mp.spawn(fn=train_fn,
        args=(world_size, configs, exp_id, g, device, train_tactile_cont_dataset, train_text_cont_dataset, exp_name),
        nprocs=world_size,
        join=True)

def cleanup():
    dist.destroy_process_group()

def prepare(dataset, rank, world_size, batch_size=64, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_encoder_epoch(rank, configs, loaders, optimizers, models, scaler=None):
    if configs["prompt_learning"]:
        models["tactile_vificlip"].train()
    else:
        models["tactile_vificlip"].eval()
        models["tactile_adapter"].train()
    models["property_classifier"].train()
    models["plain_tactile_adapter"].train()
    models["dotted_tactile_adapter"].train()
    if "property_regression" in configs["tasks"]:
        prop_reg_loader = loaders["property_regression"]
    if "tactile_contrastive" in configs["tasks"]:
        tactile_con_enum = enumerate(loaders["tactile_contrastive"])
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()
    total_prop_cls_loss, num_prop_cls_samples = 0, 0
    total_text_con_loss, num_text_con_samples = 0, 0
    total_tactile_con_loss, num_tactile_con_samples = 0, 0
    stop_text_con, stop_rgb_con, stop_tactile_con, stop_recon = False, False, False, False
    cnt = 0
    for batch in tqdm.tqdm(prop_reg_loader):
        cnt += 1
        # if rank == 0:
        if "property_regression" in configs["tasks"]:
            # Task 1: property classification
            all_tactile_frames, properties, dataset = batch
            batch_size = all_tactile_frames.shape[0]
            num_prop_cls_samples += batch_size
            # 1.1: tactile
            all_tactile_frames = all_tactile_frames.to(rank)
            tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames, None, None)
            if not configs["prompt_learning"]:
                tactile_video_features = models["tactile_adapter"](tactile_video_features)
            tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
            plain_indices = [i for i, x in enumerate(dataset) if get_dataset_sensor_type(x) == "plain"]
            plain_tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
            tactile_video_features_clone = tactile_video_features.clone()
            tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
            # 1.2: regression
            prop_preds = models["property_classifier"](tactile_video_features_clone)
            # 1.3: regression loss
            prop_cls_loss = mse_loss_fn(prop_preds, properties.to(rank))
            prop_cls_loss.backward()
            del prop_preds
            torch.cuda.empty_cache()
            total_prop_cls_loss += prop_cls_loss.item() * batch_size

        if "tactile_contrastive" in configs["tasks"]:
            try:
                # Task 2: Tactile contrastive
                _, next_tactile_con = next(tactile_con_enum)
                all_tactile_frames, datasets = next_tactile_con
                dataset = datasets[0]
                batch_size = all_tactile_frames.shape[0]
                num_tactile_con_samples += batch_size
                # 2.1: Tactile features
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(rank), None, None)
                # print(tactile_video_features.shape)
                if not configs["prompt_learning"]:
                    tactile_video_features = models["tactile_adapter"](tactile_video_features)
                if get_dataset_sensor_type(dataset) == "plain":
                    tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                elif get_dataset_sensor_type(dataset) == "dotted":
                    tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                # 2.2: Gather other tactile set(s)
                torch.cuda.set_device(rank)
                tactile_features_list = [torch.ones_like(tactile_video_features) for _ in range(dist.get_world_size())]
                dist.all_gather(tactile_features_list, tactile_video_features)
                tactile_features_list[dist.get_rank()] = tactile_video_features
                # 2.3: Contrastive loss
                cos_sim = F.cosine_similarity(tactile_features_list[0][:,None,:], tactile_features_list[1][None,:,:], dim=-1) * models["logit_scale_tactile"].module.logit_scale.exp()
                # print(models["logit_scale_tactile"].module.logit_scale.data)
                labels = torch.arange(batch_size, dtype=torch.long).to(rank)
                tactile_con_loss = (ce_loss_fn(cos_sim, labels) + ce_loss_fn(cos_sim.T, labels)) / 2
                tactile_con_loss.backward()
                del tactile_features_list
                torch.cuda.empty_cache()
                total_tactile_con_loss += tactile_con_loss.item() * batch_size
            except StopIteration:
                stop_tactile_con = True
        
        if "text_contrastive" in configs["tasks"]:
            try:
                # Task 3: text contrastive
                _, next_text_con = next(text_con_enum)
                all_tactile_frames_or_description_ids, object_type, dataset = next_text_con
                batch_size = all_tactile_frames_or_description_ids.shape[0]
                num_text_con_samples += batch_size
                data = {
                    "tactile_or_descriptions": all_tactile_frames_or_description_ids,
                    "object_type": object_type,
                    "dataset": dataset
                }
                # 3.1: Gather tactile and text
                torch.cuda.set_device(rank)
                data_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(data_list, data)
                data_list[dist.get_rank()] = data
                # NOTE: Only for 2 GPU setup
                all_tactile_frames = data_list[0]["tactile_or_descriptions"]
                description_ids = data_list[1]["tactile_or_descriptions"]
                # 3.2: Tactile
                tactile_video_features, text_features, _, _ = models["tactile_vificlip"](all_tactile_frames.to(rank), description_ids.to(rank), None)
                if not configs["prompt_learning"]:
                    tactile_video_features = models["tactile_adapter"](tactile_video_features)
                if get_dataset_sensor_type(dataset) == "plain":
                    tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                elif get_dataset_sensor_type(dataset) == "dotted":
                    tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                # 3.3: Text
                text_features = models["text_adapter"](text_features)
                # 3.4: Contrastive loss
                cos_sim = F.cosine_similarity(tactile_video_features[:,None,:], text_features[None,:,:], dim=-1) * models["logit_scale_text"].module.logit_scale.exp()
                labels = torch.arange(batch_size, dtype=torch.long).to(rank)
                text_con_loss = (ce_loss_fn(cos_sim, labels) + ce_loss_fn(cos_sim.T, labels)) / 2
                # print(text_con_loss, models["logit_scale_text"].module.logit_scale.data)
                text_con_loss.backward()
                del tactile_video_features
                del text_features
                torch.cuda.empty_cache()
                total_text_con_loss += text_con_loss.item() * batch_size
            except StopIteration:
                stop_text_con = True

        # if "rgb_contrastive" in configs["tasks"]: # FIXME
        #     try:
        #         # Task 4: RGB contrastive
        #         _, next_rgb_con = next(rgb_con_enum)
        #         all_tactile_frames, all_rgb_frames = next_rgb_con
        #         batch_size = all_tactile_frames.shape[0]
        #         num_rgb_con_samples += batch_size
        #         # 4.1: tactile
        #         tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(rank), None, None)
        #         if not configs["prompt_learning"]:
        #             tactile_video_features = models["tactile_adapter"](tactile_video_features)
        #         # 4.2: RGB
        #         rgb_video_features, _, _, _ = models["tactile_vificlip"](all_rgb_frames.to(rank), None, None)
        #         rgb_video_features = models["rgb_adapter"](rgb_video_features)
        #         # 4.3 contrastive loss
        #         cos_sim = F.cosine_similarity(tactile_video_features[:,None,:], rgb_video_features[None,:,:], dim=-1) / configs["temperature"]
        #         labels = torch.arange(batch_size, dtype=torch.long).to(rank)
        #         rgb_con_loss = (ce_loss_fn(cos_sim, labels) + ce_loss_fn(cos_sim.T, labels)) / 2
        #         total_rgb_con_loss += rgb_con_loss.item() * batch_size
        #     except StopIteration:
        #         stop_rgb_con = True

        # Total loss
        # loss = 0
        # if "property_regression" in configs["tasks"]:
        #     loss += prop_cls_loss
        # if not stop_tactile_con and "tactile_contrastive" in configs["tasks"]:
        #     loss += tactile_con_loss
        # if not stop_text_con and "text_contrastive" in configs["tasks"]:
        #     loss += text_con_loss
        # if not stop_rgb_con and "rgb_contrastive" in configs["tasks"]:
        #     loss += rgb_con_loss
        # if not stop_recon:
        #     if "reconstruction" in configs["tasks"]:
        #         loss += recon_loss
        # loss.backward()
        if configs["prompt_learning"]:
            optimizers["tactile_vificlip"][0].step()
        else:
            optimizers["tactile_adapter"][0].step()
        optimizers["plain_tactile_adapter"][0].step()
        optimizers["dotted_tactile_adapter"][0].step()
        if "property_regression" in configs["tasks"]:
            optimizers["property_classifier"][0].step()
        if "text_contrastive" in configs["tasks"] and not stop_text_con:
            optimizers["text_adapter"][0].step()
        # if "reconstruction" in configs["tasks"] and not stop_recon:
        #     optimizers["tactile_decoder"][0].step()
        if configs["prompt_learning"]:
            optimizers["tactile_vificlip"][0].zero_grad()
        else:
            optimizers["tactile_adapter"][0].zero_grad()
        optimizers["plain_tactile_adapter"][0].zero_grad()
        optimizers["dotted_tactile_adapter"][0].zero_grad()
        if "property_regression" in configs["tasks"]:
            optimizers["property_classifier"][0].zero_grad()
        if "text_contrastive" in configs["tasks"] and not stop_text_con:
            optimizers["text_adapter"][0].zero_grad()
        # if "reconstruction" in configs["tasks"] and not stop_recon:
        #     optimizers["tactile_decoder"][0].zero_grad()

    optimizers["plain_tactile_adapter"][1].step()
    optimizers["dotted_tactile_adapter"][1].step()
    optimizers["property_classifier"][1].step()
    if configs["prompt_learning"]:
        optimizers["tactile_vificlip"][1].step()
    else:
        optimizers["tactile_adapter"][1].step()
    if "text_contrastive" in configs["tasks"] and not stop_text_con:
        optimizers["text_adapter"][1].step()

    if rank == 0:
        if "tactile_contrastive" in configs["tasks"] and not stop_tactile_con:
            models["logit_scale_tactile"].module.logit_scale.data = torch.clamp(models["logit_scale_tactile"].module.logit_scale.data, 0, 4.6052)
        if "text_contrastive" in configs["tasks"] and not stop_text_con:
            models["logit_scale_text"].module.logit_scale.data = torch.clamp(models["logit_scale_text"].module.logit_scale.data, 0, 4.6052)

    results_dict = {}
    if "property_regression" in configs["tasks"]:
        results_dict["total_prop_cls_loss"] = total_prop_cls_loss
        results_dict["num_prop_cls_samples"] = num_prop_cls_samples
    if "tactile_contrastive" in configs["tasks"]:
        results_dict["total_tactile_con_loss"] = total_tactile_con_loss
        results_dict["num_tactile_con_samples"] = num_tactile_con_samples
    if "text_contrastive" in configs["tasks"]:
        results_dict["total_text_con_loss"] = total_text_con_loss
        results_dict["num_text_con_samples"] = num_text_con_samples
    # if "rgb_contrastive" in configs["tasks"]:
    #     wandb_dict["train/rgb_con_loss"] = total_rgb_con_loss / num_rgb_con_samples
    # if "reconstruction" in configs["tasks"]:
    #     wandb_dict["train/recon_loss"] = total_recon_loss / num_recon_samples
    return results_dict


def evaluate_encoder_epoch(rank, configs, loaders, models):
    models["tactile_vificlip"].eval()
    if not configs["prompt_learning"]:
        models["tactile_adapter"].eval()
    models["property_classifier"].eval()
    models["plain_tactile_adapter"].eval()
    models["dotted_tactile_adapter"].eval()
    if "property_regression" in configs["tasks"]:
        prop_reg_loader = loaders["property_regression"]
    if "text_contrastive" in configs["tasks"]:
        models["text_adapter"].eval()
    # if "rgb_contrastive" in configs["tasks"]:
    #     models["rgb_adapter"].eval()
    # if "reconstruction" in configs["tasks"]:
    #     models["tactile_decoder"].eval()
    mse_loss_fn = torch.nn.MSELoss()
    total_prop_cls_loss = 0
    num_prop_cls_samples = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(prop_reg_loader):
            if "property_regression" in configs["tasks"]:
                # Task 1: Property classification
                all_tactile_frames, properties, dataset = batch
                all_labels.append(properties.cpu().numpy())
                batch_size = all_tactile_frames.shape[0]
                num_prop_cls_samples += batch_size
                # 1.1: Tactile
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(rank), None, None)
                if not configs["prompt_learning"]:
                    tactile_video_features = models["tactile_adapter"](tactile_video_features)
                tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                plain_indices = [i for i, x in enumerate(dataset) if x == "physiclear"]
                plain_tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                tactile_video_features_clone = tactile_video_features.clone()
                tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
                # 1.2: Regression
                prop_preds = models["property_classifier"](tactile_video_features_clone)
                # 1.3: Regression loss
                prop_cls_loss = mse_loss_fn(prop_preds, properties.to(rank))
                # print(prop_preds, properties.to(rank))
                total_prop_cls_loss += prop_cls_loss.item() * batch_size
                all_preds.append(prop_preds.cpu().numpy())
        # Get classification results
        all_preds = np.concatenate(all_preds, axis=0)
        all_preds_bin = []
        for p in all_preds:
            all_preds_bin.append(np.asarray([get_categorical_labels(p[0]), get_categorical_labels(p[1])]))
        all_preds_bin = np.concatenate([all_preds_bin], axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels_bin = []
        for l in all_labels:
            all_labels_bin.append(np.asarray([get_categorical_labels(l[0]), get_categorical_labels(l[1])]))
        all_labels_bin = np.concatenate([all_labels_bin], axis=0)
        # num_samples = all_preds_bin.shape[0]
    results_dict = {
        "total_prop_cls_loss": total_prop_cls_loss,
        "hardness_correct": np.sum(all_preds_bin[:, 0] == all_labels_bin[:, 0]),
        "roughness_correct": np.sum(all_preds_bin[:, 1] == all_labels_bin[:, 1]),
        "combined_correct": np.sum(np.all(all_preds_bin == all_labels_bin, axis=-1)),
        "num_prop_cls_samples": num_prop_cls_samples,
    }
    return results_dict
    # return final_loss, results


def train_encoder(rank, world_size, configs, exp_id, g, device, train_tactile_cont_dataset, train_text_cont_dataset, exp_name):
    setup(rank, world_size)
    # Dataloaders
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(configs["use_clip"])
    train_loaders = {}
    # 1) Property regression
    train_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"])
    train_prop_reg_loader = DataLoader(train_prop_reg_dataset, batch_size=int(configs["batch_size"] / world_size), shuffle=True, generator=g, collate_fn=regression_collate_fn)
    val_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="val", datasets=configs["datasets"], frame_size=configs["frame_size"])
    val_prop_reg_loader = DataLoader(val_prop_reg_dataset, batch_size=int(configs["batch_size"] / world_size), shuffle=False, generator=g, collate_fn=regression_collate_fn)
    test_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="test", datasets=configs["datasets"], frame_size=configs["frame_size"])
    test_prop_reg_loader = DataLoader(test_prop_reg_dataset, batch_size=int(configs["batch_size"] / world_size), shuffle=False, generator=g, collate_fn=regression_collate_fn)
    train_loaders["property_regression"] = train_prop_reg_loader
    # 2) Tactile contrastive
    if "tactile_contrastive" in configs["tasks"]:
        train_tactile_cont_loader = prepare(train_tactile_cont_dataset, rank, world_size, batch_size=configs["batch_size"], pin_memory=False, num_workers=0)
        train_loaders["tactile_contrastive"] = train_tactile_cont_loader
    # 3) Text contrastive
    if "text_contrastive" in configs["tasks"]:
        train_text_cont_loader = prepare(train_text_cont_dataset, rank, world_size, batch_size=configs["batch_size"], pin_memory=False, num_workers=0)
        train_loaders["text_contrastive"] = train_text_cont_loader
    # # 4) RGB contrastive
    # if "rgb_contrastive" in configs["tasks"]:
    #     train_rgb_cont_dataset = TactileRGBContrastiveDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=configs["batch_size"])
    #     train_rgb_cont_loader = DataLoader(train_rgb_cont_dataset, batch_size=None, shuffle=True, generator=g)
    #     train_loaders["rgb_contrastive"] = train_rgb_cont_loader
    val_loaders = {
        "property_regression": val_prop_reg_loader
    }
    test_loaders = {
        "property_regression": test_prop_reg_loader
    }
    
    # Models
    # 1) Tactile
    tactile_encoder = DDP(CLIPVisionEncoder(clip_model=configs["use_clip"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    if configs["prompt_learning"]:
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs)
        if configs["gradient_checkpointing"]:
            clip.vision_model.encoder.gradient_checkpointing = True
            clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if "text_contrastive" in configs["tasks"]:
                clip.text_model.encoder.gradient_checkpointing = True
        # Save prompt learning parameters for future model loading
        prompt_learning_configs = {
            "use_clip": configs["use_clip"],
            "num_context_vision": configs["num_context_vision"],
            "prompt_depth_vision": configs["prompt_depth_vision"],
            "dim_context_vision": configs["dim_context_vision"],
            "num_context_text": configs["num_context_text"],
            "prompt_depth_text": configs["prompt_depth_text"],
            "dim_context_text": configs["dim_context_text"],
            "gate_prior": configs["gate_prior"]
        }
        prompt_learning_configs_path = f'{configs["exps_path"]}/{exp_id}/prompt_learning.yaml'
        with open(prompt_learning_configs_path, 'w') as f:
            yaml.dump(prompt_learning_configs, f)
            f.close()
    else:
        clip = CLIPModel.from_pretrained(configs["use_clip"])
        tactile_adapter = DDP(CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    plain_tactile_adapter = DDP(CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    dotted_tactile_adapter = DDP(CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    if "text_contrastive" in configs["tasks"]:
        freeze_text_encoder = False
    else:
        freeze_text_encoder = True
    tactile_vificlip = DDP(ViFiCLIP(clip, freeze_text_encoder=freeze_text_encoder, use_positional_embeds=True).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    if configs["prompt_learning"]:
        for name, param in tactile_vificlip.named_parameters():
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    logit_scale_tactile = DDP(LogitScale().to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    property_classifier = DDP(PropertyClassifier(input_size=configs["dim_context_vision"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    models = {
        "tactile_vificlip": tactile_vificlip,
        "logit_scale_tactile": logit_scale_tactile,
        "tactile_encoder": tactile_encoder,
        "property_classifier": property_classifier,
        "plain_tactile_adapter": plain_tactile_adapter,
        "dotted_tactile_adapter": dotted_tactile_adapter
    }
    if not configs["prompt_learning"]:
        models["tactile_adapter"] = tactile_adapter
    # 2) text contrastive
    if "text_contrastive" in configs["tasks"]:
        # text_encoder = CLIPTextModel.from_pretrained(configs["use_clip"]).to(device)
        # for name, param in text_encoder.named_parameters():
        #     param.requires_grad_(False)
        text_adapter = DDP(CLIPRFC(input_size=configs["dim_context_text"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
        logit_scale_text = DDP(LogitScale().to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
        models["text_adapter"] = text_adapter
        models["logit_scale_text"] = logit_scale_text
    # # 3) RGB contrastive
    # if "rgb_contrastive" in configs["tasks"]:
    #     rgb_adapter = DDP(CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #     models["rgb_adapter"] = rgb_adapter

    # Optimizers
    # 1) Tactile
    if configs["prompt_learning"]:
        tactile_vificlip_params = list(tactile_vificlip.parameters())
        if "tactile_contrastive" in configs["tasks"]:
            tactile_vificlip_params += list(logit_scale_tactile.parameters())
        if "text_contrastive" in configs["tasks"]:
            tactile_vificlip_params += list(logit_scale_text.parameters())
        optimizer_tactile_vificlip = torch.optim.AdamW(tactile_vificlip_params, lr=configs["adapter_lr"])
        scheduler_tactile_vificlip = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tactile_vificlip, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
    else:
        optimizer_tactile_adapter = torch.optim.AdamW(tactile_adapter.parameters(), lr=configs["adapter_lr"])
        scheduler_tactile_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tactile_adapter, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
    optimizer_property_classifier = torch.optim.AdamW(property_classifier.parameters(), lr=configs["adapter_lr"])
    scheduler_property_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_property_classifier, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
    optimizer_plain_tactile_adapter = torch.optim.AdamW(dotted_tactile_adapter.parameters(), lr=configs["adapter_lr"])
    scheduler_plain_tactile_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_plain_tactile_adapter, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
    optimizer_dotted_tactile_adapter = torch.optim.AdamW(dotted_tactile_adapter.parameters(), lr=configs["adapter_lr"])
    scheduler_dotted_tactile_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dotted_tactile_adapter, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
    optimizers = {
        "property_classifier": [optimizer_property_classifier, scheduler_property_classifier],
        "plain_tactile_adapter": [optimizer_plain_tactile_adapter, scheduler_plain_tactile_adapter],
        "dotted_tactile_adapter": [optimizer_dotted_tactile_adapter, scheduler_dotted_tactile_adapter],
    }
    if configs["prompt_learning"]:
        optimizers["tactile_vificlip"] = [optimizer_tactile_vificlip, scheduler_tactile_vificlip]
    else:
        optimizers["tactile_adapter"] = [optimizer_tactile_adapter, scheduler_tactile_adapter]
    if "text_contrastive" in configs["tasks"]:
        optimizer_text_adapter = torch.optim.AdamW(text_adapter.parameters(), lr=configs["adapter_lr"])
        scheduler_text_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_text_adapter, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
        optimizers["text_adapter"] = [optimizer_text_adapter, scheduler_text_adapter]

    # training
    if rank == 0 and exp_name != "debug":
        run = wandb.init(
            project="train-clip",
            name=exp_name,
            config=configs,
            # group="encoder_distributed"
        )
    best_val_loss = 99999
    epochs = configs["num_epochs"]
    if rank == 0:
        val_results, test_results = {"best": None}, {"best": None}
    scaler = GradScaler()
    for epoch in tqdm.tqdm(range(epochs)):
        if "tactile_contrastive" in configs["tasks"]:
            train_tactile_cont_loader.sampler.set_epoch(epoch)
        if "text_contrastive" in configs["tasks"]:
            train_text_cont_loader.sampler.set_epoch(epoch)
        train_results_dict = train_encoder_epoch(rank, configs, train_loaders, optimizers, models, scaler)
        val_results_dict = evaluate_encoder_epoch(rank, configs, val_loaders, models)
        test_results_dict = evaluate_encoder_epoch(rank, configs, test_loaders, models)
        print(rank, train_results_dict)
        print(rank, val_results_dict)
        # Gather results
        torch.cuda.set_device(rank)
        train_results_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(train_results_list, train_results_dict)
        train_results_list[dist.get_rank()] = train_results_dict
        val_results_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(val_results_list, val_results_dict)
        val_results_list[dist.get_rank()] = val_results_dict
        test_results_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(test_results_list, test_results_dict)
        test_results_list[dist.get_rank()] = test_results_dict
        if rank == 0:
            # Save training stats
            train_total_prop_cls_loss, train_num_prop_cls_samples = 0, 0
            # train_total_tactile_con_loss, train_num_tactile_con_samples = 0, 0
            # train_text_con_loss, train_um_text_con_samples = 0, 0
            for result_dict in train_results_list:
                train_total_prop_cls_loss += result_dict["total_prop_cls_loss"]
                train_num_prop_cls_samples += result_dict["num_prop_cls_samples"]
            # Save classification results
            val_total_prop_cls_loss, val_hardness_correct, val_roughness_correct, val_combined_correct, val_num_samples = 0, 0, 0, 0, 0
            for result_dict in val_results_list:
                val_total_prop_cls_loss += result_dict["total_prop_cls_loss"]
                val_hardness_correct += result_dict["hardness_correct"]
                val_roughness_correct += result_dict["roughness_correct"]
                val_combined_correct += result_dict["combined_correct"]
                val_num_samples += result_dict["num_prop_cls_samples"]
            val_results[epoch] = {
                "Hardness": val_hardness_correct / val_num_samples,
                "Roughness": val_roughness_correct / val_num_samples,
                "Combined": val_combined_correct / val_num_samples
            }
            val_loss = val_total_prop_cls_loss / val_num_samples
            if val_loss < best_val_loss:
                val_results["best"] = val_results[epoch].copy()
                val_results["best"]["epoch"] = epoch
            acc_json_path = f'{configs["exps_path"]}/{exp_id}/results/encoder_cls_val.json'
            with open(acc_json_path, 'w') as f:
                json.dump(val_results, f, indent=4)
                f.close()
            test_total_prop_cls_loss, test_hardness_correct, test_roughness_correct, test_combined_correct, test_num_samples = 0, 0, 0, 0, 0
            for result_dict in test_results_list:
                test_total_prop_cls_loss += result_dict["total_prop_cls_loss"]
                test_hardness_correct += result_dict["hardness_correct"]
                test_roughness_correct += result_dict["roughness_correct"]
                test_combined_correct += result_dict["combined_correct"]
                test_num_samples += result_dict["num_prop_cls_samples"]
            test_results[epoch] = {
                "Hardness": test_hardness_correct / test_num_samples,
                "Roughness": test_roughness_correct / test_num_samples,
                "Combined": test_combined_correct / test_num_samples
            }
            if val_loss < best_val_loss:
                test_results["best"] = test_results[epoch].copy()
                test_results["best"]["epoch"] = epoch
            acc_json_path = f'{configs["exps_path"]}/{exp_id}/results/encoder_cls_test.json'
            with open(acc_json_path, 'w') as f:
                json.dump(test_results, f, indent=4)
                f.close()
            if exp_name != "debug":
                wandb_dict = {
                    "train/prop_cls_loss": train_total_prop_cls_loss / train_num_prop_cls_samples,
                    "val/prop_cls_loss": val_loss,
                    "test/prop_cls_loss": test_total_prop_cls_loss / test_num_samples
                }
                if "tactile_contrastive" in configs["tasks"]:
                    wandb_dict["train/tactile_con_loss"] = train_results_list[dist.get_rank()]["total_tactile_con_loss"] / train_results_list[dist.get_rank()]["num_tactile_con_samples"]
                if "text_contrastive" in configs["tasks"]:
                    wandb_dict["train/text_con_loss"] = train_results_list[dist.get_rank()]["total_text_con_loss"] / train_results_list[dist.get_rank()]["num_text_con_samples"]
                wandb.log(wandb_dict)
            # Check if there is a new best model and save it
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tactile_encoder.module.model.vision_model = models["tactile_vificlip"].module.clip_model.vision_model
                torch.save(tactile_encoder.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_encoder.pt")
                torch.save(models["tactile_vificlip"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt")
                if not configs["prompt_learning"]:
                    torch.save(models["tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt")
                torch.save(models["property_classifier"].state_dict(), f"{configs['exps_path']}/{exp_id}/property_classifier.pt")
                torch.save(models["plain_tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/plain_tactile_adapter.pt")
                torch.save(models["dotted_tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/dotted_tactile_adapter.pt")
        dist.barrier()

    if rank == 0:
        # load best models for visualizations
        tactile_vificlip.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt"))
        if not configs["prompt_learning"]:
            tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt"))
        property_classifier.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/property_classifier.pt"))
        plain_tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/plain_tactile_adapter.pt"))
        dotted_tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/dotted_tactile_adapter.pt"))
        models["tactile_vificlip"] = tactile_vificlip
        if not configs["prompt_learning"]:
            models["tactile_adapter"] = tactile_adapter
        models["property_classifier"] = property_classifier
        models["plain_tactile_adapter"] = plain_tactile_adapter
        models["dotted_tactile_adapter"] = dotted_tactile_adapter
        os.makedirs(f"{configs['exps_path']}/{exp_id}/viz", exist_ok=True)
        visualize(configs, test_loaders, models, split="test", pca=None, device=rank, exp_id=exp_id, train=True, test=True)
        if exp_name != "debug":
            run.finish()
    dist.barrier()
    cleanup()


def visualize(configs, loaders, models, split, pca, device, exp_id, train, test):
    models["tactile_vificlip"].eval()
    # if not configs["prompt_learning"]:
    models["tactile_adapter"].eval()
    # models["plain_tactile_adapter"].eval()
    # models["dotted_tactile_adapter"].eval()
    models["property_classifier"].eval()
    num_prop_cls_samples = 0
    if "property_regression" in configs["tasks"]:
        prop_reg_loader = loaders["property_regression"]
    all_embeddings, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(prop_reg_loader):
            if "property_regression" in configs["tasks"]:
                # Task 1: Property classification
                all_tactile_frames, properties, datasets = batch
                all_labels.append(properties.cpu().numpy())
                batch_size = all_tactile_frames.shape[0]
                num_prop_cls_samples += batch_size
                # 1.1: Tactile
                sensors = [get_dataset_sensor_type(d) for d in datasets]
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(device), None, None, sensors)
                # if not configs["prompt_learning"]:
                tactile_video_features = models["tactile_adapter"](tactile_video_features)
                # tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                # plain_indices = [i for i, x in enumerate(dataset) if get_dataset_sensor_type(x) == "plain"]
                # plain_tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                # tactile_video_features_clone = tactile_video_features.clone()
                # tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
                # 1.2: Regression
                # prop_preds = models["property_classifier"](tactile_video_features_clone)
                prop_preds = models["property_classifier"](tactile_video_features)
                all_preds.append(prop_preds.cpu().numpy())
                # 1.3: Embeddings
                # all_embeddings.append(tactile_video_features_clone.cpu().numpy())
                all_embeddings.append(tactile_video_features.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels_bin = []
    for l in all_labels:
        all_labels_bin.append(np.asarray([get_categorical_labels(l[0], bins=configs["visualize_bins"]), get_categorical_labels(l[1], bins=configs["visualize_bins"])]))
    all_labels_bin = np.concatenate([all_labels_bin], axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_preds_bin = []
    for p in all_preds:
        all_preds_bin.append(np.asarray([get_categorical_labels(p[0], bins=configs["visualize_bins"]), get_categorical_labels(p[1], bins=configs["visualize_bins"])]))
    all_preds_bin = np.concatenate([all_preds_bin], axis=0)
    if not train and test:
        # 1) Classification
        num_samples = all_preds_bin.shape[0]
        hardness_acc = np.sum(all_preds_bin[:, 0] == all_labels_bin[:, 0]) / num_samples
        roughness_acc = np.sum(all_preds_bin[:, 1] == all_labels_bin[:, 1]) / num_samples
        combined_acc = np.sum(np.all(all_preds_bin == all_labels_bin, axis=-1)) / num_samples
        results = {
            "Hardness": hardness_acc,
            "Roughness": roughness_acc,
            "Combined": combined_acc
        }
        acc_json_path = f'{configs["exps_path"]}/{exp_id}/results/encoder_cls_{split}.json'
        with open(acc_json_path, 'w') as f:
            json.dump(results, f, indent=4)
            f.close()
    # 2) Confusion matrix
    labels = [i for i in range(configs["visualize_bins"])]
    hardness_order = [i for i in range(configs["visualize_bins"])]
    hardness_confusion_matrix = metrics.confusion_matrix(all_labels_bin[:, 0], all_preds_bin[:, 0], labels=labels)
    hardness_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=hardness_confusion_matrix, display_labels=hardness_order)
    hardness_cm_display.plot()
    plt.savefig(f"{configs['exps_path']}/{exp_id}/results/confusion_matrix_hardness.png")
    plt.clf()
    roughness_order = [i for i in range(configs["visualize_bins"])]
    roughness_confusion_matrix = metrics.confusion_matrix(all_labels_bin[:, 1], all_preds_bin[:, 1], labels=labels)
    rougness_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=roughness_confusion_matrix, display_labels=roughness_order)
    rougness_cm_display.plot()
    plt.savefig(f"{configs['exps_path']}/{exp_id}/results/confusion_matrix_roughness.png")
    plt.clf()
    # 3) Embeddings
    # PCA
    df = pd.DataFrame()
    df['Hardness Labels'] = [int(i) for i in all_labels_bin[:,0]]
    df['Roughness Labels'] = [int(i) for i in all_labels_bin[:,1]]
    titles = {0: "hardness", 1: "roughness"}
    orders = {0: hardness_order, 1: roughness_order}
    labels_name = {0: "Hardness Labels", 1: "Roughness Labels"}
    labels_num = {0: configs["visualize_bins"], 1: configs["visualize_bins"]}
    if pca is None:
        pca = PCA(n_components=30)
        pca.fit(all_embeddings)
    pca_result = pca.transform(all_embeddings)
    print('Cumulative explained variation for the principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    # t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    df["PCA t-SNE 1"] = tsne_results[:,0]
    df["PCA t-SNE 2"] = tsne_results[:,1]
    for label_type_idx in range(all_labels_bin.shape[1]):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="PCA t-SNE 1", y="PCA t-SNE 2",
            hue_order=orders[label_type_idx],
            hue=labels_name[label_type_idx],
            palette=sns.color_palette("hls", labels_num[label_type_idx]),
            data=df,
            legend="full",
        )
        plt.xlabel("PCA t-SNE 1", fontsize=18)
        plt.ylabel("PCA t-SNE 2", fontsize=18)
        plt.legend(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.savefig(f"{configs['exps_path']}/{exp_id}/viz/tsne_{split}_{titles[label_type_idx]}.png")
        plt.clf()


if __name__ == "__main__":
    run_type = f"run"
    config_path = f'configs/{run_type}.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    exp_name = input("\nExperiment name: ")
    if len(exp_name) == 0:
        exp_name = "debug"
    exp_id = "encoder_train"
    if len(exp_name) > 0:
        exp_id = exp_id + f"_{exp_name}"

    # Make experiment folder
    now = datetime.now()
    exp_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_id = exp_date + "_" + exp_id
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}/results", exist_ok=True)
    if configs["train_llm"] or configs["train_llm_peft"]:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/preds", exist_ok=True)
    with open(f"{configs['exps_path']}/{exp_id}/{run_type}.yaml", 'w') as file:
        documents = yaml.dump(configs, file)
        file.close()

    # Seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    # torch.use_deterministic_algorithms(True)
    random.seed(configs["seed"])
    g = torch.Generator()
    g.manual_seed(configs["seed"])
    device = f'cuda:{configs["cuda"]}' # for non-LLM models

    # Training and/or testing
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2
    world_size = n_gpus
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(configs["use_clip"])
    # 1) Tactile-tactile contrastive
    train_tactile_cont_dataset = TactileTactileContrastiveDistributedDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=configs["batch_size"])
    # 2) Tactile-text contrastive
    train_text_cont_dataset = TactileTextContrastiveDistributedDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=configs["batch_size"])
    run_fn(train_encoder, world_size, configs, exp_id, None, device, train_tactile_cont_dataset, train_text_cont_dataset, exp_name)
