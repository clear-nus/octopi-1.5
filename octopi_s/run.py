import itertools
import os
import re 
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
from utils.estimate_uncertainty import *
from utils.visualize_encoder import visualize
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
import random
import yaml
from datetime import datetime
import wandb
from scipy.stats import kendalltau


def train_encoder_epoch(configs, loaders, optimizers, models, scaler=None):
    if configs["prompt_learning"]:
        models["tactile_vificlip"].train()
    else:
        models["tactile_vificlip"].eval()
        models["tactile_adapter"].train()
    models["property_classifier"].train()
    models["plain_tactile_adapter"].train()
    models["dotted_tactile_adapter"].train()
    if "property_regression" in configs["tasks"]:
        # prop_cls_enum = enumerate(loaders["property_regression"])
        prop_reg_loader = loaders["property_regression"]
    if "tactile_contrastive" in configs["tasks"]:
        tactile_con_enum = enumerate(loaders["tactile_contrastive"])
    if "text_contrastive" in configs["tasks"]:
        text_con_enum = enumerate(loaders["text_contrastive"])
        # models["text_encoder"].eval()
        models["text_adapter"].train()
    if "rgb_contrastive" in configs["tasks"]:
        rgb_con_enum = enumerate(loaders["rgb_contrastive"])
        models["rgb_adapter"].train()
    # if "reconstruction" in configs["tasks"]:
    #     recon_enum = enumerate(loaders["reconstruction"])
    #     models["tactile_decoder"].train()
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()
    total_prop_cls_loss, num_prop_cls_samples = 0, 0
    total_text_con_loss, num_text_con_samples = 0, 0
    total_rgb_con_loss, num_rgb_con_samples = 0, 0
    total_tactile_con_loss, num_tactile_con_samples = 0, 0
    total_recon_loss, num_recon_samples = 0, 0
    stop_text_con, stop_rgb_con, stop_tactile_con, stop_recon = False, False, False, False
    for batch in tqdm.tqdm(prop_reg_loader):
        if "property_regression" in configs["tasks"]:
            # Task 1: property classification
            all_tactile_frames, properties, dataset = batch
            batch_size = all_tactile_frames.shape[0]
            num_prop_cls_samples += batch_size
            # 1.1: tactile
            all_tactile_frames = all_tactile_frames.to(device)
            tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames, None, None)
            if not configs["prompt_learning"]:
                tactile_video_features = models["tactile_adapter"](tactile_video_features)
            # if dataset == "physicleardotted" or dataset == "hardness":
            tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
            plain_indices = [i for i, x in enumerate(dataset) if get_dataset_sensor_type(x) == "plain"]
            plain_tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
            tactile_video_features_clone = tactile_video_features.clone()
            tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
            # 1.2: regression
            prop_preds = models["property_classifier"](tactile_video_features_clone)
            # 1.3: regression loss
            prop_cls_loss = mse_loss_fn(prop_preds, properties.to(device))
            total_prop_cls_loss += prop_cls_loss.item() * batch_size

        if "tactile_contrastive" in configs["tasks"]:
            try:
                # Task 2: Tactile contrastive
                _, next_tactile_con = next(tactile_con_enum)
                all_tactile_frames, all_other_tactile_frames, dataset = next_tactile_con
                batch_size = all_tactile_frames.shape[0]
                num_tactile_con_samples += batch_size
                # 2.1: tactile set 1
                tactile_video_features_1, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(device), None, None)
                if not configs["prompt_learning"]:
                    tactile_video_features_1 = models["tactile_adapter"](tactile_video_features_1)
                if get_dataset_sensor_type(dataset) == "plain":
                    tactile_video_features_1 = models["plain_tactile_adapter"](tactile_video_features_1)
                elif get_dataset_sensor_type(dataset) == "dotted":
                    tactile_video_features_1 = models["dotted_tactile_adapter"](tactile_video_features_1)
                # 2.2: tactile set 2
                tactile_video_features_2, _, _, _ = models["tactile_vificlip"](all_other_tactile_frames.to(device), None, None)
                if not configs["prompt_learning"]:
                    tactile_video_features_2 = models["tactile_adapter"](tactile_video_features_2)
                if get_dataset_sensor_type(dataset) == "plain":
                    tactile_video_features_2 = models["plain_tactile_adapter"](tactile_video_features_2)
                elif get_dataset_sensor_type(dataset) == "dotted":
                    tactile_video_features_2 = models["dotted_tactile_adapter"](tactile_video_features_2)
                # 2.3 contrastive loss
                cos_sim = F.cosine_similarity(tactile_video_features_1[:,None,:], tactile_video_features_2[None,:,:], dim=-1) * models["tactile_vificlip"].logit_scale_tactile.exp()
                labels = torch.arange(batch_size, dtype=torch.long).to(device)
                tactile_con_loss = (ce_loss_fn(cos_sim, labels) + ce_loss_fn(cos_sim.T, labels)) / 2
                total_tactile_con_loss += tactile_con_loss.item() * batch_size
            except StopIteration:
                stop_tactile_con = True
        
        if "text_contrastive" in configs["tasks"]: # FIXME
            try:
                # Task 3: text contrastive
                _, next_text_con = next(text_con_enum)
                all_tactile_frames, description_ids, dataset = next_text_con
                batch_size = all_tactile_frames.shape[0]
                num_text_con_samples += batch_size
                # 3.1: tactile
                tactile_video_features, text_features, _, _ = models["tactile_vificlip"](all_tactile_frames.to(device), description_ids.to(device), None)
                if not configs["prompt_learning"]:
                    tactile_video_features = models["tactile_adapter"](tactile_video_features)
                if get_dataset_sensor_type(dataset) == "plain":
                    tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                elif get_dataset_sensor_type(dataset) == "dotted":
                    tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                # 3.2: text
                # text_features = models["text_encoder"](input_ids=description_ids.to(device)).pooler_output
                text_features = models["text_adapter"](text_features)
                # 3.3 contrastive loss
                cos_sim = F.cosine_similarity(tactile_video_features[:,None,:], text_features[None,:,:], dim=-1) * models["tactile_vificlip"].logit_scale_text.exp()
                labels = torch.arange(batch_size, dtype=torch.long).to(device)
                text_con_loss = (ce_loss_fn(cos_sim, labels) + ce_loss_fn(cos_sim.T, labels)) / 2
                total_text_con_loss += text_con_loss.item() * batch_size
            except StopIteration:
                stop_text_con = True

        # total loss
        loss = 0
        if "property_regression" in configs["tasks"]:
            loss += prop_cls_loss
        if not stop_tactile_con and "tactile_contrastive" in configs["tasks"]:
            loss += tactile_con_loss
        if not stop_text_con and "text_contrastive" in configs["tasks"]:
            loss += text_con_loss
        loss.backward()
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
        if "reconstruction" in configs["tasks"] and not stop_recon:
            optimizers["tactile_decoder"][0].step()
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
        if "reconstruction" in configs["tasks"] and not stop_recon:
            optimizers["tactile_decoder"][0].zero_grad()
    optimizers["plain_tactile_adapter"][1].step()
    optimizers["dotted_tactile_adapter"][1].step()
    optimizers["property_classifier"][1].step()
    if configs["prompt_learning"]:
        optimizers["tactile_vificlip"][1].step()
    else:
        optimizers["tactile_adapter"][1].step()
    if "text_contrastive" in configs["tasks"] and not stop_text_con:
            optimizers["text_adapter"][1].step()
    
    if "tactile_contrastive" in configs["tasks"] and not stop_tactile_con:
        models["tactile_vificlip"].logit_scale_tactile.data = torch.clamp(models["tactile_vificlip"].logit_scale_tactile.data, 0, 4.6052)
    if "text_contrastive" in configs["tasks"] and not stop_text_con:
        models["tactile_vificlip"].logit_scale_text.data = torch.clamp(models["tactile_vificlip"].logit_scale_text.data, 0, 4.6052)
        
    wandb_dict = {}
    if "property_regression" in configs["tasks"]:
        wandb_dict["train/prop_cls_loss"] = total_prop_cls_loss / num_prop_cls_samples
    if "text_contrastive" in configs["tasks"]:
        wandb_dict["train/text_con_loss"] = total_text_con_loss / num_text_con_samples
    if "rgb_contrastive" in configs["tasks"]:
        wandb_dict["train/rgb_con_loss"] = total_rgb_con_loss / num_rgb_con_samples
    if "tactile_contrastive" in configs["tasks"]:
        wandb_dict["train/tactile_con_loss"] = total_tactile_con_loss / num_tactile_con_samples
    # if "reconstruction" in configs["tasks"]:
    #     wandb_dict["train/recon_loss"] = total_recon_loss / num_recon_samples
    return wandb_dict


def evaluate_encoder_epoch(configs, loaders, models, results, epoch, exp_id, split=None):
    models["tactile_vificlip"].eval()
    if not configs["prompt_learning"]:
        models["tactile_adapter"].eval()
    models["property_classifier"].eval()
    models["plain_tactile_adapter"].eval()
    models["dotted_tactile_adapter"].eval()
    if "property_regression" in configs["tasks"]:
        prop_reg_loader = loaders["property_regression"]
    if "text_contrastive" in configs["tasks"]:
        # models["text_encoder"].eval()
        models["text_adapter"].eval()
    if "rgb_contrastive" in configs["tasks"]:
        models["rgb_adapter"].eval()
    if "reconstruction" in configs["tasks"]:
        models["tactile_decoder"].eval()
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
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(device), None, None)
                if not configs["prompt_learning"]:
                    tactile_video_features = models["tactile_adapter"](tactile_video_features)
                tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                plain_indices = [i for i, x in enumerate(dataset) if get_dataset_sensor_type(x) == "plain"]
                plain_tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                tactile_video_features_clone = tactile_video_features.clone()
                tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
                # 1.2: Regression
                prop_preds = models["property_classifier"](tactile_video_features_clone)
                # 1.3: Regression loss
                prop_cls_loss = mse_loss_fn(prop_preds, properties.to(device))
                total_prop_cls_loss += prop_cls_loss.item() * batch_size
                all_preds.append(prop_preds.cpu().numpy())
        final_loss = total_prop_cls_loss / num_prop_cls_samples
        # Save classification results
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
        num_samples = all_preds_bin.shape[0]
        hardness_acc = np.sum(all_preds_bin[:, 0] == all_labels_bin[:, 0]) / num_samples
        roughness_acc = np.sum(all_preds_bin[:, 1] == all_labels_bin[:, 1]) / num_samples
        combined_acc = np.sum(np.all(all_preds_bin == all_labels_bin, axis=-1)) / num_samples
        results[epoch] = {
            "Hardness": hardness_acc,
            "Roughness": roughness_acc,
            "Combined": combined_acc
        }
        acc_json_path = f'{configs["exps_path"]}/{exp_id}/results/encoder_cls_{split}.json'
        with open(acc_json_path, 'w') as f:
            json.dump(results, f, indent=4)
            f.close()
    return final_loss, {f"{split}/prop_cls_loss": final_loss}, results


def run_encoder(configs, exp_id, g, device, train, test):
    # Dataloaders
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(configs["use_clip"])
    if train:
        train_loaders = {}
        # 1) Property regression
        train_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"])
        train_prop_reg_loader = DataLoader(train_prop_reg_dataset, batch_size=configs["batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g, collate_fn=regression_collate_fn)
        val_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="val", datasets=configs["datasets"], frame_size=configs["frame_size"])
        val_prop_reg_loader = DataLoader(val_prop_reg_dataset, batch_size=configs["batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g, collate_fn=regression_collate_fn)
        train_loaders["property_regression"] = train_prop_reg_loader
    test_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="test", datasets=configs["datasets"], frame_size=configs["frame_size"])
    test_prop_reg_loader = DataLoader(test_prop_reg_dataset, batch_size=configs["batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g, collate_fn=regression_collate_fn)
    if train:
    # 2) Text contrastive
        if "text_contrastive" in configs["tasks"]:
            train_text_cont_dataset = TactileTextContrastiveDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=configs["batch_size"])
            train_text_cont_loader = DataLoader(train_text_cont_dataset, batch_size=None, shuffle=True, worker_init_fn=seed_worker, generator=g)
            train_loaders["text_contrastive"] = train_text_cont_loader
        # 3) RGB contrastive
        if "rgb_contrastive" in configs["tasks"]:
            train_rgb_cont_dataset = TactileRGBContrastiveDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=configs["batch_size"])
            train_rgb_cont_loader = DataLoader(train_rgb_cont_dataset, batch_size=None, shuffle=True, worker_init_fn=seed_worker, generator=g)
            train_loaders["rgb_contrastive"] = train_rgb_cont_loader
        # 4) Tactile contrastive
        if "tactile_contrastive" in configs["tasks"]:
            train_tactile_cont_dataset = TactileTactileContrastiveDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=configs["batch_size"])
            train_tactile_cont_loader = DataLoader(train_tactile_cont_dataset, batch_size=None, shuffle=True, worker_init_fn=seed_worker, generator=g)
            train_loaders["tactile_contrastive"] = train_tactile_cont_loader
        val_loaders = {"property_regression": val_prop_reg_loader}
    test_loaders = {"property_regression": test_prop_reg_loader}
    
    # Models
    # 1) Tactile
    tactile_encoder = CLIPVisionEncoder(clip_model=configs["use_clip"]).to(device)
    if configs["prompt_learning"]:
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
        if configs["gradient_checkpointing"]:
            clip.vision_model.encoder.gradient_checkpointing = True
            clip.gradient_checkpointing_enable()
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
        clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
        tactile_adapter = CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(device)
    if "text_contrastive" in configs["tasks"]:
        freeze_text_encoder = False
    else:
        freeze_text_encoder = True
    tactile_vificlip = ViFiCLIP(clip, freeze_text_encoder=freeze_text_encoder, use_positional_embeds=True).to(device)
    if configs["load_exp_path"] is not None:
        if os.path.exists(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt")):
            tactile_vificlip.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt"), map_location=device, weights_only=True), strict=False)
            print("Loaded tactile ViFi-CLIP!")
    if train:
        if configs["prompt_learning"]:
            for name, param in tactile_vificlip.named_parameters():
                if "VPT" in name or "logit_scale" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        else:
            for name, param in tactile_vificlip.named_parameters():
                if "logit_scale" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    plain_tactile_adapter = CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(device)
    if configs["load_exp_path"] is not None:
        if os.path.exists(os.path.join(configs["load_exp_path"], "plain_tactile_adapter.pt")):
            plain_tactile_adapter.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "plain_tactile_adapter.pt"), map_location=device, weights_only=True))
            print("Loaded plain_tactile_adapter!")
    dotted_tactile_adapter = CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(device)
    if configs["load_exp_path"] is not None:
        if os.path.exists(os.path.join(configs["load_exp_path"], "dotted_tactile_adapter.pt")):
            dotted_tactile_adapter.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "dotted_tactile_adapter.pt"), map_location=device, weights_only=True))
            print("Loaded dotted_tactile_adapter!")
    property_classifier = PropertyClassifier(input_size=configs["dim_context_vision"]).to(device)
    if configs["load_exp_path"] is not None:
        if os.path.exists(os.path.join(configs["load_exp_path"], "property_classifier.pt")):
            property_classifier.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "property_classifier.pt"), map_location=device, weights_only=True))
            print("Loaded property classifier!")
    models = {
        "tactile_vificlip": tactile_vificlip,
        "tactile_encoder": tactile_encoder,
        "property_classifier": property_classifier,
        "plain_tactile_adapter": plain_tactile_adapter,
        "dotted_tactile_adapter": dotted_tactile_adapter
    }
    if not configs["prompt_learning"]:
        models["tactile_adapter"] = tactile_adapter

    # 2) Text contrastive
    if "text_contrastive" in configs["tasks"]:
        text_encoder = CLIPTextModel.from_pretrained(configs["use_clip"]).to(device)
        if train:
            for name, param in text_encoder.named_parameters():
                param.requires_grad_(False)
        text_adapter = CLIPRFC(input_size=configs["dim_context_text"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(device)
        models["text_encoder"] = text_encoder
        models["text_adapter"] = text_adapter
        
    # # 3) RGB contrastive
    # if "rgb_contrastive" in configs["tasks"]:
    #     rgb_adapter = CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"], residual_ratio=configs["residual_ratio"]).to(device)
    #     models["rgb_adapter"] = rgb_adapter

    if train:
        # optimizers
        # 1) Tactile
        if configs["prompt_learning"]:
            optimizer_tactile_vificlip = torch.optim.AdamW(tactile_vificlip.parameters(), lr=configs["adapter_lr"])
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
        # 2) Text contrastive
        if "text_contrastive" in configs["tasks"]:
            # optimizer_text_encoder = torch.optim.AdamW(text_encoder.parameters(), lr=configs["adapter_lr"])
            optimizer_text_adapter = torch.optim.AdamW(text_adapter.parameters(), lr=configs["adapter_lr"])
            # optimizers["text_encoder"] = optimizer_text_encoder
            scheduler_text_adapter =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_text_adapter, T_max=configs["num_epochs"], eta_min=configs["adapter_lr"]/10)
            optimizers["text_adapter"] = [optimizer_text_adapter, scheduler_text_adapter]
        # 3) RGB contrastive
        if "rgb_contrastive" in configs["tasks"]:
            optimizer_rgb_adapter = torch.optim.AdamW(rgb_adapter.parameters(), lr=configs["adapter_lr"])
            optimizers["rgb_adapter"] = [optimizer_rgb_adapter]
        # # 4) Reconstruction
        # if "reconstruction" in configs["tasks"]:
            # optimizer_tactile_decoder = torch.optim.AdamW(tactile_decoder.parameters(), lr=configs["adapter_lr"])
            # optimizers["tactile_decoder"] = optimizer_tactile_decoder #, scheduler_tactile_decoder],

    # Training
    if train:
        best_val_loss = 99999
        epochs = configs["num_epochs"]
        val_results, test_results = {"best": None}, {"best": None}
        # scaler = GradScaler()
        for epoch in tqdm.tqdm(range(epochs)):
            wandb_dict = train_encoder_epoch(configs, train_loaders, optimizers, models, scaler=None)
            val_loss, val_loss_dict, val_results = evaluate_encoder_epoch(configs, val_loaders, models, val_results, epoch, exp_id, split="val")
            _, test_loss_dict, test_results = evaluate_encoder_epoch(configs, test_loaders, models, test_results, epoch, exp_id, split="test")
            wandb_dict.update(val_loss_dict)
            wandb_dict.update(test_loss_dict)
            try:
                wandb.log(wandb_dict)
            except wandb.errors.Error:
                pass
            if val_loss < best_val_loss:
                val_results["best"] = val_results[epoch].copy()
                val_results["best"]["epoch"] = epoch
                test_results["best"] = test_results[epoch].copy()
                test_results["best"]["epoch"] = epoch
                best_val_loss = val_loss
                tactile_encoder.model.vision_model = models["tactile_vificlip"].clip_model.vision_model
                torch.save(tactile_encoder.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_encoder.pt")
                torch.save(models["tactile_vificlip"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt")
                if not configs["prompt_learning"]:
                    torch.save(models["tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt")
                torch.save(models["property_classifier"].state_dict(), f"{configs['exps_path']}/{exp_id}/property_classifier.pt")
                torch.save(models["plain_tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/plain_tactile_adapter.pt")
                torch.save(models["dotted_tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/dotted_tactile_adapter.pt")

    # Visualize
    if train:
        tactile_vificlip.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt"))
        if not configs["prompt_learning"]:
            tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt"))
        property_classifier.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/property_classifier.pt"))
        plain_tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/plain_tactile_adapter.pt"))
        dotted_tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/dotted_tactile_adapter.pt"))
        models["tactile_vificlip"] = tactile_vificlip
        if not configs["prompt_learning"]:
            models["tactile_adapter"] = tactile_adapter
        models["plain_tactile_adapter"] = plain_tactile_adapter
        models["dotted_tactile_adapter"] = dotted_tactile_adapter
        models["property_classifier"] = property_classifier
    os.makedirs(f"{configs['exps_path']}/{exp_id}/viz", exist_ok=True)
    visualize(configs, test_loaders, models, split="test", pca=None, device=device, exp_id=exp_id, train=train, test=test)



def run_llm(configs, exp_id, g, device, train, peft):
    # load tokenizer and LLM weights
    tokenizer_path, model_path, new_tokens, no_split_module_classes = get_model_details(configs["model_type"])
    os.makedirs(configs["offload_dir"], exist_ok=True)
    f = open(configs["gpu_config"])
    gpu_config = json.load(f)
    model = load_mllm(configs, tokenizer_path, model_path, new_tokens, no_split_module_classes, peft, device, gpu_config, exp_id=None)
    tokenizer = model.tokenizer

    # RAG
    if configs["rag"]:
        if "prompt_learning.yaml" in os.listdir(configs["load_exp_path"]):
            prompt_learning_configs = yaml.safe_load(open(os.path.join(configs["load_exp_path"], "prompt_learning.yaml")))
            clip = PromptLearningCLIPModel.from_pretrained(prompt_learning_configs["use_clip"], prompt_learning_configs).to(device)
        else:
            clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
        tactile_vificlip = ViFiCLIP(clip, freeze_text_encoder=True, use_positional_embeds=True).to(device)
        if os.path.exists(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt")):
            tactile_vificlip.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt"), map_location=device, weights_only=True), strict=False)
            print("Loaded tactile ViFi-CLIP for RAG!")
        tactile_vificlip.eval()
        saved_embeddings, sample_tactile_paths, object_ids = get_rag_train_embeddings(tactile_vificlip, configs, device)
    else:
        tactile_vificlip = None
        saved_embeddings = None
        sample_tactile_paths = None
        object_ids = None

    # Load datasets
    if configs["use_clip"]:
        image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    test, reason = False, False
    if train:
        if len(configs["train_files"]) > 0:
            train_dataset = TactileLLMDataset(image_processor, configs["train_files"], split_name="train", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"])
            train_loader = DataLoader(train_dataset, batch_size=configs["per_device_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if train:
        if len(configs["test_files"]) > 0:
            test = True
            test_dataset = TactileLLMDataset(image_processor, configs["test_files"], split_name="test", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"])
            test_loader = DataLoader(test_dataset, batch_size=configs["per_device_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    if len(configs["reasoning_files"]) > 0:
        reason = True
        reasoning_dataset = TactileLLMDataset(image_processor, configs["reasoning_files"], split_name="test", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"], rag=configs["rag"], tactile_vificlip=tactile_vificlip, saved_embeddings=saved_embeddings, sample_tactile_paths=sample_tactile_paths, object_ids=object_ids, device=device, retrieval_object_num=configs["retrieval_object_num"])
        reasoning_loader = DataLoader(reasoning_dataset, batch_size=configs["per_device_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    # Training parameters
    # 1) LLM
    if train:
        llm_params = []
        if not peft:
            for name, param in model.llm.named_parameters():
                # NOTE: no lm_head here since they are not tied to word embeddings in LLaMA and no new tokens for generation
                if "vicuna" in configs["model_type"]:
                    if "lora" in name or "embed_tokens" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                if param.requires_grad:
                    llm_params.append(param)
        else:
            for name, param in model.llm.named_parameters():
                if param.requires_grad:
                    llm_params.append(param)
        if len(llm_params) > 0 and len(configs["train_files"]) > 0:
            optimizer_llm = torch.optim.AdamW(llm_params, lr=configs["llm_lr"])
            num_steps = int(len(train_loader) / configs["llm_gradient_accumulation_steps"])
            scheduler_llm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=num_steps)

    # 2) Encoder
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    for name, param in model.tactile_adapter.named_parameters():
        param.requires_grad = False
    for name, param in model.plain_tactile_adapter.named_parameters():
        param.requires_grad = False
    for name, param in model.dotted_tactile_adapter.named_parameters():
        param.requires_grad = False

    # 3) Projection
    for name, param in model.project.named_parameters():
        param.requires_grad = not configs["freeze_projection"]
    if not configs["freeze_projection"]:
        project_params = model.project.parameters()
        if train and len(configs["train_files"]) > 0:
            optimizer_project = torch.optim.AdamW(project_params, lr=configs["projection_lr"])
            scheduler_project = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_project, T_max=len(train_loader) / configs["llm_gradient_accumulation_steps"])

    # Training
    if train and len(configs["train_files"]) > 0:
        model.train()
        model.encoder.eval()
        model.tactile_adapter.eval()
        # get trainable/non-trainable model parameter stats
        trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        all_params = sum([np.prod(p.size()) for p in model.parameters()])
        if peft:
            max_train_steps = configs["max_peft_train_steps"]
        else:
            max_train_steps = configs["max_train_steps"]
        if max_train_steps < len(train_loader):
            print(f"\nFinetuning LLM for {max_train_steps} samples and {int(max_train_steps / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        else:
            print(f"\nFinetuning LLM for {len(train_loader)} samples and {int(len(train_loader) / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        print('Trainable params: {} ({:.2f}%)'.format(trainable_params, trainable_params / all_params * 100,))
        # total_train_loss = 0
        for train_sample_step, batch in enumerate(t:=tqdm.tqdm(train_loader)):
            question, chat, answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict = batch
            answer_tokens = answer_tokens.to(device)
            outputs, _ = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_datasets=all_datasets, all_indices=all_indices)
            train_loss = outputs.loss.detach().float()
            t.set_description(f"Train loss: {train_loss}")
            loss = outputs.loss / configs["llm_gradient_accumulation_steps"]
            loss.backward()
            if (train_sample_step + 1) % configs["llm_gradient_accumulation_steps"] == 0:
                # optimizer updates
                if not configs["freeze_projection"]:
                    optimizer_project.step()
                    scheduler_project.step()
                    optimizer_project.zero_grad()
                if len(llm_params) > 0:
                    optimizer_llm.step()
                    scheduler_llm.step()
                    optimizer_llm.zero_grad()
            if (train_sample_step + 1) >= max_train_steps:
                break
        print("Saving tokenizer and models...")
        tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_id}/tokenizer")
        if peft:
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_id}/llm_weights_peft")
        else:
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_id}/llm_weights")
        torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_encoder.pt")
        torch.save(model.tactile_adapter.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt")
        torch.save(model.plain_tactile_adapter.state_dict(), f"{configs['exps_path']}/{exp_id}/plain_tactile_adapter.pt")
        torch.save(model.dotted_tactile_adapter.state_dict(), f"{configs['exps_path']}/{exp_id}/dotted_tactile_adapter.pt")
        torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_id}/project.pt")
        print(f"LLM finetuning done!")

    # Testing
    if test:
        print(f"\nTesting LLM on the test set...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        preds = []
        beam_dict = {}
        with torch.no_grad():
            for test_sample_step, batch in enumerate(tqdm.tqdm(test_loader)):
                all_objects, sample_paths = [], []
                # NOTE: hardcoded for batch size of 1
                question, chat, answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict = batch
                answer_tokens = answer_tokens.to(device)
                if configs["ranking_sampling"] and "rank" in question[0].lower():
                    # Get all object permutations
                    question_str = chat[0]["content"][0]
                    task = question_str.split("Object")[0] + "Object"
                    objects = question_str.split("Object")[1:]
                    objects = [i.replace("\n", "") for i in objects]
                    num_objects = len(objects)
                    object_permutations = list(itertools.permutations(objects))
                    selected_object_permutations_hardness = []
                    selected_object_permutations_roughness = []
                    for i in range(configs["ranking_sampling_num"]):
                        object_permutation = random.choice(object_permutations)
                        # Permute tactile_frames and all_datasets to match object_permutation
                        indices = [objects.index(obj) for obj in object_permutation]
                        permuted_tactile_frames = [tactile_frames[i] for i in indices]
                        permuted_all_datasets = [all_datasets[i] for i in indices]
                        new_question = task + "Object".join([obj + "\n\n" for o, obj in enumerate(object_permutation)])
                        new_question = new_question.strip("\n")
                        permuted_question = chat[:-1]
                        permuted_question[0]["content"] = new_question
                        permuted_question[0]["role"] = "user"
                        permuted_question = [model.tokenizer.apply_chat_template(permuted_question, tokenize=False, add_generation_prompt=True)]
                        _, question_embeds = model(question=permuted_question, tactile_frames=permuted_tactile_frames, answer_tokens=answer_tokens, all_datasets=permuted_all_datasets, all_indices=all_indices, question_embeds_only=True)
                        generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None)
                        generation_i = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                        generation_i = generation_i.strip().split(tokenizer.eos_token)[0].strip()
                        if "decreasing" not in generation_i:
                            continue
                        rankings_i = generation_i.split("decreasing")[1:]
                        rankings_i = [re.sub(r"[^\d.,]", "", r.strip()).split(",") for r in rankings_i]
                        print(rankings_i)
                        if len(rankings_i[0]) == num_objects:
                            selected_object_permutations_hardness.append(rankings_i[0])
                        if len(rankings_i[1]) == num_objects:
                            selected_object_permutations_roughness.append(rankings_i[1])
                    tau_sum_hardness = {}
                    print(selected_object_permutations_hardness, selected_object_permutations_roughness)
                    for i, object_permutation_i in enumerate(selected_object_permutations_hardness):
                        tau_sum_hardness[i] = 0
                        for j, object_permutation_j in enumerate(selected_object_permutations_hardness):
                            if i != j:
                                tau, _ = kendalltau(object_permutation_i, object_permutation_j)
                                kendall_tau_distance = 1 - tau
                                tau_sum_hardness[i] += kendall_tau_distance
                    tau_sum_roughness = {}
                    for i, object_permutation_i in enumerate(selected_object_permutations_roughness):
                        tau_sum_roughness[i] = 0
                        for j, object_permutation_j in enumerate(selected_object_permutations_roughness):
                            if i != j:
                                tau, _ = kendalltau(object_permutation_i, object_permutation_j)
                                kendall_tau_distance = 1 - tau
                                tau_sum_roughness[i] += kendall_tau_distance
                    try:
                        best_hardness_ranking = min(tau_sum_hardness, key=tau_sum_hardness.get)
                        best_roughness_ranking = min(tau_sum_roughness, key=tau_sum_roughness.get)
                    except ValueError:
                        pass
                _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None)
                # # TODO: Estimate sorting uncertainty
                # first_position_pairwise_dict = get_first_position_pairwise(model, question_embeds, configs, all_objects_dict, tokenizer)
                # beam_dict = get_beam_pairwise_and_sequence(model, question_embeds, configs, all_objects_dict, tokenizer)
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                answer_tokens = answer_tokens[0].cpu().numpy()
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                if configs["ranking_sampling"] and "rank" in question[0].lower():
                    print(generation)
                    # TODO: Check if ranking exists in generation
                    # TODO: Permute tactile_frames and all_datasets to match object_permutation
                for i in tactile:
                    sample_paths.append(i[0])
                    data = json.load(open(os.path.join("/".join(i[0].split("/")[:-1]), "data.json"), "r"))
                    all_objects.append(data["object_id"])
                preds.append({
                    "sample_paths": sample_paths,
                    "all_objects": all_objects,
                    "question": question,
                    "final_true_answer": answer,
                    "final_generation": generation,
                    # "first_position_pairwise": first_position_pairwise_dict,
                    # "beam": beam_dict,
                })
            if peft:
                preds_json_path = f'{configs["exps_path"]}/{exp_id}/preds/llm_peft.json'
            else:
                preds_json_path = f'{configs["exps_path"]}/{exp_id}/preds/llm.json'
            with open(preds_json_path, 'w') as f:
                json.dump(preds, f, indent=4)
                f.close()
        print(f"LLM testing done!")

    # Reasoning
    if reason:
        print(f"\nRunning LLM on the reasoning set...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        # FIXME: Classifier loading is not proper
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
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
        freeze_text_encoder = True
        tactile_vificlip = ViFiCLIP(clip, freeze_text_encoder=freeze_text_encoder, use_positional_embeds=True).to(device)
        property_classifier = PropertyClassifier(input_size=configs["dim_context_vision"]).to(device)
        if configs["load_exp_path"] is not None:
            if os.path.exists(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt")):
                tactile_vificlip.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt"), map_location=device, weights_only=True), strict=False)
                print("Loaded tactile ViFi-CLIP!")
            if os.path.exists(os.path.join(configs["load_exp_path"], "property_classifier.pt")):
                property_classifier.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "property_classifier.pt"), map_location=device, weights_only=True))
                print("Loaded property classifier!")
        mse_loss_fn = torch.nn.MSELoss()
        # FIXME: Classifier loading is not proper
        all_reason = {}
        sample_no = {}
        with torch.no_grad():
            for reasoning_sample_step, batch in enumerate(tqdm.tqdm(reasoning_loader)):
                all_objects, sample_paths = [], []
                # NOTE: hardcoded for batch size of 1
                chat, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict, scenario, scenario_steps, target, num_candidates, rag_outputs = batch
                generated_chat = []
                scenario = f"{scenario[0]}_{target[0]}"
                check_scenario = False
                if scenario_steps.item() == 0:
                    continue
                if configs["scenarios"] is not None:
                    for scenario_to_check in configs["scenarios"]:
                        if scenario_to_check in scenario:
                            check_scenario = True
                            break
                    if not check_scenario:
                        continue
                if scenario not in all_reason.keys():
                    all_reason[scenario] = []
                    sample_no[scenario] = 1
                else:
                    sample_no[scenario] += 1
                if configs["user_stop_idx"] is not None:
                    chat = chat[:int(configs["user_stop_idx"]*2)]
                for c in range(len(chat)-1):
                    # NOTE: Only for batch size = 1
                    chat[c] = {k:v[0] for k,v in chat[c].items()}
                    if c % 2 == 0:
                        # Question
                        # NOTE: Only for batch size = 1
                        generated_chat.append(chat[c])
                    else:
                        # Answer
                        answer_idx = int((c-1)/2)
                        # RAG and change descriptions
                        if answer_idx in configs["generate_idx"]:
                            # Generate
                            question = [tokenizer.apply_chat_template(generated_chat, tokenize=False, add_generation_prompt=True)]
                            _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                            # if answer_idx == 0 and configs["description_sampling_num"] > 1:
                            #     generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=True, temperature=configs["description_temperature"], num_return_sequences=configs["description_sampling_num"], top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                            #     description_counts = {}
                            #     for seq in generation_tokens.sequences:
                            #         generation = tokenizer.decode(seq, skip_special_tokens=True).strip()
                            #         generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                            #         descriptions = generation.split(": ")[-1].split(", ")
                            #         # Get rid of full stop
                            #         for description in descriptions:
                            #             description = description.replace(".", "")
                            #             if description not in description_counts.keys():
                            #                 description_counts[description] = 1
                            #             else:
                            #                 description_counts[description] += 1
                            #     new_descriptions = []
                            #     for description, cnt in description_counts.items():
                            #         if cnt / configs["description_sampling_num"] >= configs["description_threshold"]:
                            #             new_descriptions.append(description)
                            #     generation = generation.split(": ")[0] + ": " + ", ".join(sorted(new_descriptions)) + "."
                            #     print(target, description_counts, generation)
                            # 1) Descriptions
                            generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                            generation = tokenizer.decode(generation_tokens.sequences[0], skip_special_tokens=True).strip()
                            generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                            chat[c]["generate"] = True
                            chat[c]["true_answer"] = chat[c]["content"]
                            chat[c]["content"] = generation
                            generated_chat.append(chat[c])
                        else:
                            chat[c]["generate"] = False
                            generated_chat.append(chat[c])
                        # NOTE: Only for one touched object in the guessing game
                        if answer_idx == 0 and configs["rag"]: # NOTE: Assume generate_idx=0 is the description
                            chat[c]["content"] += "\nMost similar objects (in order of decreasing similarity):"
                            for obj_name, obj_descriptions in rag_outputs[0].items():
                                # chat[c]["content"] += f" {obj_name} ({', '.join(sorted([i[0] for i in obj_descriptions]))});"
                                chat[c]["content"] += f" {obj_name};"
                            chat[c]["content"] = chat[c]["content"][:-1] # NOTE: Remove last comma
                final_question = [tokenizer.apply_chat_template(generated_chat, tokenize=False, add_generation_prompt=True)]
                final_true_answer = chat[-1]["content"][0]
                _, question_embeds = model(question=final_question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                if configs["reasoning_sampling_num"] == 1:
                    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                    final_generation = tokenizer.decode(generation_tokens.sequences[0], skip_special_tokens=True).strip()
                    final_generation = final_generation.strip().split(tokenizer.eos_token)[0].strip()
                else:
                    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=True, temperature=configs["reasoning_temperature"], num_return_sequences=configs["reasoning_sampling_num"], top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                    option_generations = {}
                    option_counts = {}
                    option_entropies = {}
                    if configs["reasoning_selection_type"] == "best_of_n":
                        entropies = get_sentence_entropy(generation_tokens, token_start_index=0)
                        max_avg_entropy_per_token = max([i["avg_entropy_per_token"] for i in entropies])
                    for seq_idx, seq in enumerate(generation_tokens.sequences):
                        generation = tokenizer.decode(seq, skip_special_tokens=True).strip()
                        generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                        option = generation.replace("*", "").split("Answer: ")[-1][0]
                        if option not in ["A", "B", "C"]:
                            # NOTE: max_new_tokens is probably not high enough so these are not present in the answer or answer is formatted incorrectly (rare)
                            print(generation)
                            continue
                        if option not in option_generations.keys():
                            option_generations[option] = [generation]
                            option_counts[option] = 1
                            if configs["reasoning_selection_type"] == "best_of_n":
                                option_entropies[option] = [(max_avg_entropy_per_token - entropies[seq_idx]["avg_entropy_per_token"]) / max_avg_entropy_per_token]
                        else:
                            option_generations[option].append(generation)
                            option_counts[option] += 1
                            if configs["reasoning_selection_type"] == "best_of_n":
                                option_entropies[option].append((max_avg_entropy_per_token - entropies[seq_idx]["avg_entropy_per_token"]) / max_avg_entropy_per_token)
                    if configs["reasoning_selection_type"] == "majority_voting":
                        # Get random generation from best option
                        most_common_option = max(option_counts, key=option_counts.get)
                        final_generation = random.choice(option_generations[most_common_option])
                    elif configs["reasoning_selection_type"] == "best_of_n":
                        # Weigh with average entropy per token
                        best_option = max(option_entropies, key=lambda k: sum(option_entropies[k]))
                        final_generation = option_generations[best_option][option_entropies[best_option].index(max(option_entropies[best_option]))]
                # if final_true_answer.split("Object ")[1][0] == final_generation.split("Object ")[1][0]:
                #     correct = True
                # else:
                #     correct = False
                # if final_true_answer[0] == final_generation[0]:
                #     correct = True
                # else:
                #     correct = False
                # # 1) Get white-box guessing uncertainties
                # guess_uncertainty_stats, object_scores_dict = get_guess_stats(model.tokenizer, generation_tokens, len(tactile))
                # 2) Get linguistic confidence scores
                # top_option_probabilities = get_linguistic_confidence(model, configs, tokenizer, generated_chat, tactile_frames, all_datasets, all_indices)
                # best_generation_answer = best_generation.split("Answer: ")[-1][0]
                # # 3) Get regression errors
                # all_prop_preds = []
                # all_prop_cls_loss = []
                # for i in tactile:
                #     sample_paths.append(i[0])
                #     data = json.load(open(os.path.join("/".join(i[0].split("/")[:-1]), "data.json"), "r"))
                #     all_objects.append(data["object_id"])
                # for i in range(len(tactile)):
                #     data = json.load(open(os.path.join("/".join(tactile[i][0].split("/")[:-1]), "data.json"), "r"))
                #     tactile_video_features, _, _, _ = tactile_vificlip(tactile_frames[i].to(device), None, None)
                #     # if not configs["prompt_learning"]:
                #     #     tactile_video_features = models["tactile_adapter"](tactile_video_features)
                #     tactile_video_features = model.dotted_tactile_adapter(tactile_video_features)
                #     plain_indices = [i for i, x in enumerate(all_datasets[i][0]) if get_dataset_sensor_type(x) == "plain"]
                #     plain_tactile_video_features = model.plain_tactile_adapter(tactile_video_features)
                #     tactile_video_features_clone = tactile_video_features.clone()
                #     tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
                #     prop_preds = property_classifier(tactile_video_features_clone)
                #     properties = torch.from_numpy(np.array([[data["properties"]["hardness"], data["properties"]["roughness"]]]))
                #     prop_cls_loss = mse_loss_fn(prop_preds, properties.to(device))
                #     all_prop_preds.append(prop_preds[0].cpu().numpy().tolist())
                #     all_prop_cls_loss.append(prop_cls_loss.item())
                # reverse_vocab = {v: k for k, v in model.tokenizer.vocab.items()}
                # if "llama-3" in reasoning_loader.dataset.model_type:
                #     answer_tokens = encode_text(tokenizer, final_true_answer + reasoning_loader.dataset.eot_token)
                # else:
                #     answer_tokens = encode_text(tokenizer, final_true_answer + reasoning_loader.dataset.eos_token)
                # answer_tokens = answer_tokens.to(device)
                # target_object = int(reverse_vocab[answer_tokens[2]])
                all_reason[scenario].append({
                    "sample_no": sample_no[scenario],
                    "sample_paths": sample_paths,
                    "all_objects": all_objects,
                    "num_candidates": num_candidates.item(),
                    "chat": generated_chat,
                    "generate_idx": configs["generate_idx"],
                    "user_stop_idx": configs["user_stop_idx"],
                    "description_sampling_num": configs["description_sampling_num"],
                    "reasoning_sampling_num": configs["reasoning_sampling_num"],
                    "reasoning_selection_type": configs["reasoning_selection_type"],
                    "final_true_answer": final_true_answer,
                    "final_generation": final_generation,
                    "option_counts": option_counts,
                    "option_entropies": {k: sum(v) for k, v in option_entropies.items()},
                    # "top_option_probabilities": top_option_probabilities,
                    # "entropy": guess_uncertainty_stats["entropy"],
                    # "max_prob": guess_uncertainty_stats["max_prob"].astype(float),
                    # "max_diff": guess_uncertainty_stats["max_diff"].astype(float),
                    # "mse": all_prop_cls_loss,
                    # "average_mse": sum(all_prop_cls_loss) / len(all_prop_cls_loss),
                    # "chosen_mse": all_prop_cls_loss[target_object-1],
                    # "correct": correct,
                    # "all_prop_preds": all_prop_preds
                })
            # NOTE: Prioritize PEFT LLM?
            if os.path.exists(os.path.join(configs["load_exp_path"], "llm_weights_peft")) or os.path.exists(os.path.join(f'{configs["exps_path"]}/{exp_id}/', "llm_weights_peft")):
                peft = True
            else:
                peft = False
            # Save predictions by scenario
            for scenario in all_reason.keys():
                if peft:
                    reason_json_path = f'{configs["exps_path"]}/{exp_id}/reason/{scenario}_peft.json'
                else:
                    reason_json_path = f'{configs["exps_path"]}/{exp_id}/reason/{scenario}.json'
                with open(reason_json_path, 'w') as f:
                    json.dump(all_reason[scenario], f, indent=4)
                    f.close()
        print(f"LLM reasoning done!")
    
    # Clean up
    del model
    with torch.no_grad():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_type = f"run"
    config_path = f'configs/{run_type}.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    exp_name = input("\nExperiment name: ")
    if len(exp_name) == 0:
        exp_name = "debug"
    exp_id = ""
    if configs["train_encoder"]:
        exp_id += "_train_encoder"
    elif configs["test_encoder"]:
        exp_id += "_test_encoder"
    if configs["train_llm"] and len(configs["train_files"]) > 0:
        exp_id += "_train_llm"
    if configs["train_llm_peft"] and len(configs["train_files"]) > 0:
        exp_id += "_train_peft"
    if len(configs["test_files"]) > 0:
        exp_id += "_test"
    if configs["reason_llm"]:
        exp_id += "_reason"
    if len(exp_name) > 0:
        exp_id = exp_id + f"_{exp_name}"

    # Make experiment folder
    now = datetime.now()
    exp_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_id = exp_date + exp_id
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}", exist_ok=True)
    if configs["train_encoder"] or configs["test_encoder"]:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/results", exist_ok=True)
    if configs["train_llm"] or configs["train_llm_peft"]:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/preds", exist_ok=True)
    if configs["reason_llm"]:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/reason", exist_ok=True)
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
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])
    device = f'cuda:{configs["cuda"]}' # for non-LLM models

    # Training and/or testing
    if configs["train_encoder"]:
        if exp_name != "debug":
            run = wandb.init(
                project="train-clip",
                name=exp_name,
                config=configs
            )
    if configs["train_encoder"] or configs["test_encoder"]:
        run_encoder(configs, exp_id, g, device, train=configs["train_encoder"], test=configs["test_encoder"])
        configs["load_exp_path"] = f"{configs['exps_path']}/{exp_id}"
    if configs["train_encoder"]:
        run.finish()
    train_llm = False
    if configs["train_llm"]:
        train_llm = True
        run_llm(configs, exp_id, g, device, train=train_llm, peft=False)
        configs["load_exp_path"] = f"{configs['exps_path']}/{exp_id}"
    if configs["train_llm_peft"]:
        train_llm = True
        run_llm(configs, exp_id, g, device, train=train_llm, peft=True)
    if not train_llm and configs["reason_llm"]:
        run_llm(configs, exp_id, g, device, train=train_llm, peft=False)