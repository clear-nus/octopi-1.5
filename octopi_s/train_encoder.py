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
import random, math
import yaml
from datetime import datetime
from transformers import CLIPImageProcessor, AutoConfig, AutoModelForCausalLM, AutoTokenizer
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


def setup(rank, world_size):
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_fn(train_fn, world_size, configs, exp_id, g, device, train_tactile_cont_datasets, train_text_cont_datasets, exp_name, dataset_to_object_num_map, datasets_to_use):
    mp.spawn(fn=train_fn,
        args=(world_size, configs, exp_id, g, device, train_tactile_cont_datasets, train_text_cont_datasets, exp_name, dataset_to_object_num_map, datasets_to_use),
        nprocs=world_size,
        join=True)

def cleanup():
    dist.destroy_process_group()

def prepare(dataset, rank, world_size, batch_size, shuffle, pin_memory=False, num_workers=0, generator=None, collate_fn=None):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    if generator is None or collate_fn is None:
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=sampler, generator=generator, collate_fn=collate_fn)
    return dataloader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_encoder_epoch(rank, configs, loaders, optimizers, models, datasets_to_use, scaler=None):
    if configs["prompt_learning"]:
        models["tactile_vificlip"].train()
    else:
        models["tactile_vificlip"].eval()
    models["tactile_adapter"].train()
    if "property_regression" in configs["tasks"]:
        models["property_classifier"].train()
        prop_reg_loader = loaders["property_regression"]
    if "tactile_contrastive" in configs["tasks"]:
        models["tactile_contrastive"].train()
        tactile_con_enums = {}
        for dataset_name in loaders["tactile_contrastive"].keys():
            tactile_con_enums[dataset_name] = enumerate(loaders["tactile_contrastive"][dataset_name])
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()
    stop_tactile_con = False
    total_prop_cls_loss, num_prop_cls_samples = 0, 0
    total_tactile_con_loss, num_tactile_con_samples = 0, 0
    
    cnt = 0
    for batch in tqdm.tqdm(prop_reg_loader):
        dataset_to_use = datasets_to_use[cnt]
        cnt += 1
        if "property_regression" in configs["tasks"]:
            # Task 1: property classification
            all_tactile_frames, properties, datasets = batch
            batch_size = all_tactile_frames[0].shape[0]
            num_prop_cls_samples += batch_size
            # 1.1: tactile
            all_tactile_frames = all_tactile_frames[0].to(rank) # (B, L, 3, 224, 224)
            sensors = [get_dataset_sensor_type(d) for d in datasets]
            tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames, None, None, sensors)
            tactile_video_features = models["tactile_adapter"](tactile_video_features)
            # 1.2: regression
            prop_preds = models["property_classifier"](tactile_video_features)
            # 1.3: regression loss
            prop_cls_loss = mse_loss_fn(prop_preds, properties.to(rank))
            print(f"Property regression loss: {prop_cls_loss}")
            prop_cls_loss.backward()
            del prop_preds
            torch.cuda.empty_cache()
            total_prop_cls_loss += prop_cls_loss.item() * batch_size

        if "tactile_contrastive" in configs["tasks"]:
            try:
                # Task 2: Tactile contrastive
                _, next_tactile_con = next(tactile_con_enums[dataset_to_use])
                print(f"Dataset to use: {dataset_to_use}")
                all_tactile_frames, datasets = next_tactile_con
                print(rank, datasets[0], set(datasets))
                batch_size = all_tactile_frames.shape[0]
                num_tactile_con_samples += batch_size
                # 2.1: total_tactile_con_loss
                sensors = [get_dataset_sensor_type(d) for d in datasets]
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(rank), None, None, sensors)
                tactile_video_features = models["tactile_adapter"](tactile_video_features)
                tactile_video_features = models["tactile_contrastive"](tactile_video_features)
                # 2.2: Gather other tactile set(s)
                torch.cuda.set_device(rank)
                tactile_features_list = [torch.ones_like(tactile_video_features) for _ in range(dist.get_world_size())]
                dist.all_gather(tactile_features_list, tactile_video_features)
                tactile_features_list[dist.get_rank()] = tactile_video_features
                # 2.3: Contrastive loss
                cos_sim = F.cosine_similarity(tactile_features_list[0][:,None,:], tactile_features_list[1][None,:,:], dim=-1) * models["logit_scale_tactile"].module.logit_scale.exp()
                labels = torch.arange(batch_size, dtype=torch.long).to(rank)
                tactile_con_loss = (ce_loss_fn(cos_sim, labels) + ce_loss_fn(cos_sim.T, labels)) / 2
                print(f"Tactile-tactile contrastive loss: {tactile_con_loss}")
                tactile_con_loss.backward()
                del tactile_features_list
                torch.cuda.empty_cache()
                total_tactile_con_loss += tactile_con_loss.item() * batch_size
            except StopIteration:
                stop_tactile_con = True
        optimizers["clip"][0].step()
        optimizers["clip"][0].zero_grad()
        optimizers["tasks"][0].step()
        optimizers["tasks"][0].zero_grad()

    optimizers["clip"][1].step()
    optimizers["tasks"][1].step()
    if rank == 0:
        if "tactile_contrastive" in configs["tasks"] and not stop_tactile_con:
            models["logit_scale_tactile"].module.logit_scale.data = torch.clamp(models["logit_scale_tactile"].module.logit_scale.data, 0, 4.6052)

    results_dict = {}
    if "property_regression" in configs["tasks"]:
        results_dict["total_prop_cls_loss"] = total_prop_cls_loss
        results_dict["num_prop_cls_samples"] = num_prop_cls_samples
    if "tactile_contrastive" in configs["tasks"]:
        results_dict["total_tactile_con_loss"] = total_tactile_con_loss
        results_dict["num_tactile_con_samples"] = num_tactile_con_samples
    return results_dict


def evaluate_encoder_epoch(rank, configs, loaders, models):
    models["tactile_vificlip"].eval()
    models["tactile_adapter"].eval()
    if "property_regression" in configs["tasks"]:
        models["property_classifier"].eval()
        prop_reg_loader = loaders["property_regression"]
    mse_loss_fn = torch.nn.MSELoss()
    total_prop_cls_loss = 0
    num_prop_cls_samples = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(prop_reg_loader):
            if "property_regression" in configs["tasks"]:
                # Task 1: Property classification
                all_tactile_frames, properties, datasets = batch
                all_labels.append(properties.cpu().numpy())
                batch_size = all_tactile_frames[0].shape[0]
                num_prop_cls_samples += batch_size
                # 1.1: Tactile
                all_tactile_frames = all_tactile_frames[0].to(rank) # (B, L, 3, 224, 224)
                sensors = [get_dataset_sensor_type(d) for d in datasets]
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames, None, None, sensors)
                tactile_video_features = models["tactile_adapter"](tactile_video_features)
                # 1.2: Regression
                prop_preds = models["property_classifier"](tactile_video_features)
                # 1.3: Regression loss
                prop_cls_loss = mse_loss_fn(prop_preds, properties.to(rank))
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
    results_dict = {
        "total_prop_cls_loss": total_prop_cls_loss,
        "hardness_correct": np.sum(all_preds_bin[:, 0] == all_labels_bin[:, 0]),
        "roughness_correct": np.sum(all_preds_bin[:, 1] == all_labels_bin[:, 1]),
        "combined_correct": np.sum(np.all(all_preds_bin == all_labels_bin, axis=-1)),
        "num_prop_cls_samples": num_prop_cls_samples,
    }
    return results_dict


def train_encoder(rank, world_size, configs, exp_id, g, device, train_tactile_cont_datasets, train_text_cont_datasets, exp_name, dataset_to_object_num_map, datasets_to_use):
    setup(rank, world_size)
    # Dataloaders
    image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(configs["use_clip"])
    train_loaders = {}
    # 1) Property regression
    train_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=configs["datasets"], frame_size=configs["frame_size"], flip_p=configs["flip_p"])
    train_prop_reg_loader = prepare(train_prop_reg_dataset, rank, world_size, batch_size=configs["batch_size"], shuffle=True, pin_memory=False, num_workers=0, generator=g, collate_fn=regression_collate_fn)
    # train_prop_reg_loader = DataLoader(train_prop_reg_dataset, batch_size=int(configs["batch_size"] / world_size), shuffle=True, generator=g, collate_fn=regression_collate_fn)
    val_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="val", datasets=configs["datasets"], frame_size=configs["frame_size"])
    val_prop_reg_loader = prepare(val_prop_reg_dataset, rank, world_size, batch_size=configs["batch_size"], shuffle=False, pin_memory=False, num_workers=0, generator=g, collate_fn=regression_collate_fn)
    # val_prop_reg_loader = DataLoader(val_prop_reg_dataset, batch_size=int(configs["batch_size"] / world_size), shuffle=False, generator=g, collate_fn=regression_collate_fn)
    test_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="test", datasets=configs["datasets"], frame_size=configs["frame_size"])
    test_prop_reg_loader = prepare(test_prop_reg_dataset, rank, world_size, batch_size=configs["batch_size"], shuffle=False, pin_memory=False, num_workers=0, generator=g, collate_fn=regression_collate_fn)
    # test_prop_reg_loader = DataLoader(test_prop_reg_dataset, batch_size=int(configs["batch_size"] / world_size), shuffle=False, generator=g, collate_fn=regression_collate_fn)
    train_loaders["property_regression"] = train_prop_reg_loader
    # 2) Tactile contrastive
    if "tactile_contrastive" in configs["tasks"]:
        train_tactile_cont_loaders = {}
        for dataset_name, dataset in train_tactile_cont_datasets.items():
            train_tactile_cont_loaders[dataset_name] = prepare(train_tactile_cont_datasets[dataset_name], rank, world_size, batch_size=dataset_to_object_num_map[dataset_name], shuffle=False, pin_memory=False, num_workers=0)
        train_loaders["tactile_contrastive"] = train_tactile_cont_loaders
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
        sensors = sorted(list(set([get_dataset_sensor_type(d) for d in dataset_to_object_num_map.keys()])))
        clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs, sensors)
        if configs["gradient_checkpointing"]:
            clip.vision_model.encoder.gradient_checkpointing = True
            clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Save prompt learning parameters for future model loading
        prompt_learning_configs = {
            "use_clip": configs["use_clip"],
            "num_context_vision": configs["num_context_vision"],
            "prompt_depth_vision": configs["prompt_depth_vision"],
            "dim_context_vision": configs["dim_context_vision"],
            "num_context_text": configs["num_context_text"],
            "num_context_sensor": configs["num_context_sensor"],
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
    freeze_text_encoder = True
    tactile_vificlip = DDP(ViFiCLIP(clip, freeze_text_encoder=freeze_text_encoder, use_positional_embeds=True).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    if configs["prompt_learning"]:
        for name, param in tactile_vificlip.named_parameters():
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    tactile_adapter = DDP(CLIPRFC(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    property_classifier = DDP(PropertyClassifier(input_size=configs["dim_context_vision"]).to(rank), device_ids=[rank], 
    output_device=rank, find_unused_parameters=True)
    if "tactile_contrastive" in configs["tasks"]:
        logit_scale_tactile = DDP(LogitScale().to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
        tactile_contrastive = DDP(ContrastiveAdapter(input_size=configs["dim_context_vision"], output_size=configs["dim_context_vision"]).to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=True)
    models = {
        "tactile_vificlip": tactile_vificlip,
        "logit_scale_tactile": logit_scale_tactile,
        "tactile_encoder": tactile_encoder,
        "tactile_adapter": tactile_adapter,
        "property_classifier": property_classifier,
        "tactile_contrastive": tactile_contrastive,
    }
    # Get relevant sensor adapters
    used_datasets = []
    for k, v in train_loaders.items():
        if type(v) == dict:
            for kk, vv in v.items():
                used_datasets += vv.dataset.all_datasets
        else:
            used_datasets += v.dataset.all_datasets
    used_datasets = sorted(list(set(used_datasets)))

    # Optimizers
    # 1) Tactile
    clip_params = []
    task_params = []
    if configs["prompt_learning"]:
        clip_params += list(tactile_vificlip.parameters())
        if "tactile_contrastive" in configs["tasks"]:
            clip_params += list(logit_scale_tactile.parameters())
        # optimizer_tactile_vificlip = torch.optim.AdamW(tactile_vificlip_params, lr=configs["task_lr"])
        # scheduler_tactile_vificlip = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tactile_vificlip, T_max=configs["num_epochs"], eta_min=configs["task_lr"]/10)
    task_params += list(tactile_adapter.parameters())
    if "tactile_contrastive" in configs["tasks"]:
        task_params += list(tactile_contrastive.parameters())
    task_params += list(property_classifier.parameters())
    optimizer_clip = torch.optim.AdamW(clip_params, lr=configs["clip_lr"])
    scheduler_clip = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_clip, T_max=configs["num_epochs"], eta_min=configs["clip_lr"]/10)
    optimizer_tasks = torch.optim.AdamW(task_params, lr=configs["task_lr"])
    scheduler_tasks = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tasks, T_max=configs["num_epochs"], eta_min=configs["task_lr"]/10)
    optimizers = {
        "clip": [optimizer_clip, scheduler_clip],
        "tasks": [optimizer_tasks, scheduler_tasks]
    }

    # training
    if rank == 0 and exp_name != "debug":
        run = wandb.init(
            project="train-clip",
            name=exp_name,
            config=configs,
        )
    best_val_loss = 99999
    epochs = configs["num_epochs"]
    if rank == 0:
        val_results, test_results = {"best": None}, {"best": None}
    scaler = GradScaler()
    for epoch in tqdm.tqdm(range(epochs)):
        if "tactile_contrastive" in configs["tasks"]:
            for dataset_name in train_loaders["tactile_contrastive"].keys():
                train_loaders["tactile_contrastive"][dataset_name].sampler.set_epoch(epoch)
        train_results_dict = train_encoder_epoch(rank, configs, train_loaders, optimizers, models, datasets_to_use, scaler)
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
                wandb.log(wandb_dict)
            # Check if there is a new best model and save it
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tactile_encoder.module.model.vision_model = models["tactile_vificlip"].module.clip_model.vision_model
                torch.save(tactile_encoder.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_encoder.pt")
                torch.save(models["tactile_vificlip"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt")
                torch.save(models["tactile_adapter"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt")
                torch.save(models["property_classifier"].state_dict(), f"{configs['exps_path']}/{exp_id}/property_classifier.pt")
                if "tactile_contrastive" in configs["tasks"]:
                    torch.save(models["tactile_contrastive"].state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_contrastive.pt")
        dist.barrier()

    if rank == 0:
        # load best models for visualizations
        tactile_vificlip.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/tactile_vificlip.pt"))
        tactile_adapter.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt"))
        property_classifier.load_state_dict(torch.load(f"{configs['exps_path']}/{exp_id}/property_classifier.pt"))
        models["tactile_adapter"] = tactile_adapter
        models["tactile_vificlip"] = tactile_vificlip
        models["property_classifier"] = property_classifier
        os.makedirs(f"{configs['exps_path']}/{exp_id}/viz", exist_ok=True)
        visualize(configs, test_loaders, models, split="test", pca=None, device=rank, exp_id=exp_id, train=True, test=True)
        if exp_name != "debug":
            run.finish()
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    run_type = f"run"
    config_path = f'configs/{run_type}.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    exp_name = input("\nExperiment name: ")
    if len(exp_name) == 0:
        exp_name = "debug"
    exp_id = "train_encoder"
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
    # Get number of unique training objects with >=2 samples for each dataset
    dataset_to_object_num_map = {}
    dataset_objects = {}
    for sample in os.listdir(configs["data_dir"]):
        dataset = sample.split("_")[0]
        if dataset not in configs["datasets"]:
            continue
        if dataset not in dataset_objects.keys():
            dataset_objects[dataset] = {}
        data = json.load(open(os.path.join(configs["data_dir"], sample + "/data.json"), "r"))
        if data["split"] != "train":
            continue
        if "object_id" in data.keys():
            if data["object_id"] not in dataset_objects[dataset].keys():
                dataset_objects[dataset][data["object_id"]] = 1
            else:
                dataset_objects[dataset][data["object_id"]] += 1
    for dataset, object_counts in dataset_objects.items():
        dataset_objects[dataset] = {k for k, v in object_counts.items() if v >= 2} # NOTE: >=2 samples
        dataset_to_object_num_map[dataset] = min(len(dataset_objects[dataset]), 32)
    # Get sequence of datasets manually based on number of steps in property regression dataset
    num_distributed_contrastive_batches = configs["num_distributed_contrastive_batches"]
    datasets_to_use = []
    for _ in range(num_distributed_contrastive_batches):
        datasets_to_use.append(random.choice(list(dataset_to_object_num_map.keys())))
    train_tactile_cont_datasets, train_text_cont_datasets = None, None
    print(dataset_to_object_num_map)
    if "tactile_contrastive" in configs["tasks"]:
        # 1) Tactile-tactile contrastive
        train_tactile_cont_datasets = {}
        for dataset, num_objects in dataset_to_object_num_map.items():
            train_tactile_cont_datasets[dataset] = TactileTactileContrastiveDistributedDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="train", datasets=[dataset], frame_size=configs["frame_size"], flip_p=configs["flip_p"], batch_size=dataset_to_object_num_map[dataset], num_distributed_contrastive_batches=configs["num_distributed_contrastive_batches"])
    # Run
    run_fn(train_encoder, world_size, configs, exp_id, None, device, train_tactile_cont_datasets, train_text_cont_datasets, exp_name, dataset_to_object_num_map, datasets_to_use)