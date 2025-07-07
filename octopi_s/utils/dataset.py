import pickle 
import torch 
from torch.utils.data import Dataset
import numpy as np
import os
import natsort
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import random
import json
from .physiclear_constants import *
from torchvision.transforms.functional import crop
from typing import TypeVar, Optional, Iterator
T_co = TypeVar('T_co', covariant=True)


def regression_collate_fn(data):
    # Images
    max_frame_length = 0
    frames = []
    properties = []
    datasets = []
    for k in data:
        # frames.append(torch.squeeze(torch.stack(k[0]), dim=0))
        frame_len = torch.squeeze(torch.stack(k[0]), dim=0).shape[0]
        if frame_len > max_frame_length:
            max_frame_length = frame_len
        properties.append(k[1])
        datasets.append(k[2])
    for k in data:
        new_k = torch.squeeze(torch.stack(k[0]), dim=0)
        frame_len = new_k.shape[0]
        if frame_len < max_frame_length:
            new_k_0 = torch.stack([new_k[0]] * (max_frame_length - frame_len), dim=0)
            new_k = torch.cat([new_k_0, new_k], dim=0)
        frames.append(new_k)
    frames = torch.stack(frames)
    properties = torch.stack(properties)
    return frames, properties, datasets
    

class TactilePropertyRegressionDataset(Dataset):
    def __init__(self, image_processor, tokenizer, data_path, split_name, datasets, frame_size, flip_p=0):
        super().__init__()
        self.split_name = split_name
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_samples = 0
        self.tactile = {}
        self.properties = {}
        self.datasets = []
        self.frame_size = frame_size
        for sample in os.listdir(data_path):
            sample_dataset = sample.split("_")[0]
            try:
                data = json.load(open(os.path.join(data_path, sample + "/data.json"), "r"))
            except FileNotFoundError:
                continue
            if split_name != data['split']:
                continue
            if "tactile" not in os.listdir(os.path.join(data_path, sample)):
                continue
            if "properties" in data.keys():
                if sample.split("_")[0] not in datasets:
                    continue
                if sample_dataset not in self.datasets:
                    self.datasets.append(sample_dataset)
                    self.properties[sample_dataset] = []
                    self.tactile[sample_dataset] = []
                self.num_samples += 1
                self.tactile[sample_dataset].append(os.path.join(data_path, sample + "/tactile"))
                self.properties[sample_dataset].append([data['properties']['hardness'], data['properties']['roughness']])
    
    def __len__(self): 
        return self.num_samples

    def __getitem__(self, index):
        # 1) Choose dataset
        dataset = random.choice(self.datasets)
        index = index % len(self.tactile[dataset])
        # 2) Get tactile data
        image_transforms = get_image_transforms(self.frame_size, dataset, split_name=self.split_name, flip_p=self.flip_p)
        # 2) Get tactile data
        tactile = self.tactile[dataset][index]
        all_tactile_frames = []
        if self.split_name == "train":
            tactile_frames, _ = get_frames(tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        else:
            tactile_frames, _ = get_frames(tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        all_tactile_frames.append(tactile_frames) # [(l, c, h, w)]
        # 3) Get property labels
        properties = torch.Tensor(self.properties[dataset][index])
        return all_tactile_frames, properties, dataset
    

class TactileTactileContrastiveDataset(Dataset):
    def __init__(self, image_processor, tokenizer, data_path, split_name, datasets, frame_size, flip_p=0, batch_size=None):
        super().__init__()
        self.split_name = split_name
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.object_id = {}
        self.tactile = {}
        self.tactile_by_object_id = {}
        self.datasets = []
        self.frame_size = frame_size
        self.num_samples = 0
        for sample in os.listdir(data_path):
            sample_dataset = sample.split("_")[0]
            try:
                data = json.load(open(os.path.join(data_path, sample + "/data.json"), "r"))
            except FileNotFoundError:
                continue
            if split_name != data['split']:
                continue
            if "tactile" not in os.listdir(os.path.join(data_path, sample)):
                continue
            if sample_dataset not in datasets:
                continue
            if "schaeffler" in sample_dataset:
                continue
            if sample_dataset not in self.datasets:
                self.datasets.append(sample_dataset)
                self.object_id[sample_dataset] = []
                self.tactile[sample_dataset] = []
                self.tactile_by_object_id[sample_dataset] = {}
            self.num_samples += 1
            self.object_id[sample_dataset].append(data["object_id"])
            if data["object_id"] not in self.tactile_by_object_id[sample_dataset].keys():
                self.tactile_by_object_id[sample_dataset][data["object_id"]] = [os.path.join(data_path, sample + "/tactile")]
            else:
                self.tactile_by_object_id[sample_dataset][data["object_id"]].append(os.path.join(data_path, sample + "/tactile"))
            self.tactile[sample_dataset].append(os.path.join(data_path, sample + "/tactile"))
    
    def __len__(self): 
        return self.num_samples

    def __getitem__(self, index):
        # 1) Choose dataset
        dataset = random.choice(self.datasets)
        index = index % len(self.tactile[dataset])
        # 2) Get tactile data
        # transforms_list = [
        #     transforms.ToTensor(),
        #     transforms.Resize(self.frame_size, interpolation=3),
        #     transforms.Normalize(
        #         mean=[0.48145466, 0.4578275, 0.40821073],
        #         std=[0.26862954, 0.26130258, 0.27577711]
        #     )
        # ]
        # if self.split_name == "train":
        #     if random.random() < self.flip_p:
        #         transforms_list.append(transforms.RandomHorizontalFlip(1))
        #     if random.random() < self.flip_p:
        #         transforms_list.append(transforms.RandomVerticalFlip(1))
        # transforms_list.append(transforms.CenterCrop(self.frame_size))
        # image_transforms = transforms.Compose(transforms_list)
        mean, std = get_dataset_img_norm(dataset)
        transforms_list = [transforms.Resize(self.frame_size, interpolation=3)]
        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        else:
            transforms_list.append(transforms.CenterCrop(self.frame_size))
        image_transforms = transforms.Compose(transforms_list)
        valid_current_object_id = False
        current_object_id = self.object_id[dataset][index]
        # Skip for cases where an object ID only has one tactile sample
        while not valid_current_object_id:
            try:
                tactile_set = random.sample(self.tactile_by_object_id[dataset][current_object_id], 2)
                valid_current_object_id = True
            except ValueError:
                index += 1
                current_object_id = self.object_id[dataset][index]
        tactile_anchor = tactile_set[0]
        if self.split_name == "train":
            tactile_frames, _ = get_frames(tactile_anchor, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        else:
            tactile_frames, _ = get_frames(tactile_anchor, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        all_tactile_frames = [tactile_frames]
        similar_tactile = tactile_set[1]
        if self.split_name == "train":
            tactile_frames, _ = get_frames(similar_tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        else:
            tactile_frames, _ = get_frames(similar_tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        all_other_tactile_frames = [tactile_frames]
        # 3) get dissimilar tactile data
        object_ids = list(self.tactile_by_object_id[dataset].keys())
        random.shuffle(object_ids)
        for object_id in object_ids:
            if object_id != current_object_id:
                try:
                    tactile_set = random.sample(self.tactile_by_object_id[dataset][object_id], 2)
                except ValueError:
                    continue
                if self.split_name == "train":
                    tactile_frames, _ = get_frames(tactile_set[0], self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
                else:
                    tactile_frames, _ = get_frames(tactile_set[0], self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
                all_tactile_frames.append(tactile_frames)
                if self.split_name == "train":
                    tactile_frames, _ = get_frames(tactile_set[1], self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
                else:
                    tactile_frames, _ = get_frames(tactile_set[1], self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
                all_other_tactile_frames.append(tactile_frames)
            if len(all_tactile_frames) >= self.batch_size:
                break
        num_objects = len(all_tactile_frames)
        max_frame_length = 0
        for i in range(num_objects):
            frame_len = all_tactile_frames[i].shape[0]
            if frame_len > max_frame_length:
                max_frame_length = frame_len
            frame_len = all_other_tactile_frames[i].shape[0]
            if frame_len > max_frame_length:
                max_frame_length = frame_len
        for i in range(num_objects):
            # Tactile set 1
            tactile = all_tactile_frames[i]
            frame_len = tactile.shape[0]
            if frame_len < max_frame_length:
                new_tactile = torch.stack([all_tactile_frames[i][0]] * (max_frame_length - frame_len), dim=0)
                all_tactile_frames[i] = torch.cat([new_tactile, all_tactile_frames[i]], dim=0)
            # Tactile set 2
            tactile = all_other_tactile_frames[i]
            frame_len = tactile.shape[0]
            if frame_len < max_frame_length:
                new_tactile = torch.stack([all_other_tactile_frames[i][0]] * (max_frame_length - frame_len), dim=0)
                all_other_tactile_frames[i] = torch.cat([new_tactile, all_other_tactile_frames[i]], dim=0)
        all_tactile_frames = torch.stack(all_tactile_frames)
        all_other_tactile_frames = torch.stack(all_other_tactile_frames)
        return all_tactile_frames, all_other_tactile_frames, dataset


class TactileTactileContrastiveDistributedDataset(Dataset):
    def __init__(self, image_processor, tokenizer, data_path, split_name, datasets, frame_size, flip_p=0, batch_size=None):
        super().__init__()
        self.split_name = split_name
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.object_id = {}
        self.tactile = {}
        self.tactile_by_object_id = {}
        self.datasets = []
        self.frame_size = frame_size
        self.num_samples = 0
        self.all_tactile = []
        self.all_datasets = []
        self.skipped_datasets = {}
        for sample in os.listdir(data_path):
            sample_dataset = sample.split("_")[0]
            try:
                data = json.load(open(os.path.join(data_path, sample + "/data.json"), "r"))
            except FileNotFoundError:
                continue
            if split_name != data['split'] or "tactile" not in os.listdir(os.path.join(data_path, sample)) or sample_dataset not in datasets:
                continue
            if sample_dataset not in self.datasets:
                self.datasets.append(sample_dataset)
                self.object_id[sample_dataset] = []
                self.tactile[sample_dataset] = []
                self.tactile_by_object_id[sample_dataset] = {}
            self.num_samples += 1
            # Save object ID depending on dataset
            self.object_id[sample_dataset].append(data["object_id"])
            # Save tactile depending on dataset and object ID
            if data["object_id"] not in self.tactile_by_object_id[sample_dataset].keys():
                self.tactile_by_object_id[sample_dataset][data["object_id"]] = [os.path.join(data_path, sample + "/tactile")]
            else:
                self.tactile_by_object_id[sample_dataset][data["object_id"]].append(os.path.join(data_path, sample + "/tactile"))
            # Save tactile depending on dataset
            self.tactile[sample_dataset].append(os.path.join(data_path, sample + "/tactile"))

        # Fill up samples based on batch size
        num_batches = 100
        for i in range(num_batches):
            # 1) Choose dataset
            dataset = random.choice(self.datasets)
            index = random.randint(0, len(self.object_id[dataset])-1)
            valid_current_object_id = False
            current_object_id = self.object_id[dataset][index]
            while not valid_current_object_id:
                # 2) Get tactile data
                # Skip for cases where an object ID only has one tactile sample
                try:
                    tactile_set = random.sample(self.tactile_by_object_id[dataset][current_object_id], 2)
                    valid_current_object_id = True
                except ValueError:
                    index += 1
                    current_object_id = self.object_id[dataset][index]
            # NOTE: Only for 2 GPU setup
            tactile_batch = [tactile_set[0], tactile_set[1]]
            # 3) Get dissimilar tactile data
            object_ids = list(self.tactile_by_object_id[dataset].keys())
            random.shuffle(object_ids)
            for object_id in object_ids:
                if object_id != current_object_id:
                    try:
                        tactile_set = random.sample(self.tactile_by_object_id[dataset][object_id], 2)
                    except ValueError:
                        continue
                    # NOTE: Only for 2 GPU setup
                    tactile_batch.append(tactile_set[0])
                    tactile_batch.append(tactile_set[1])
                if len(tactile_batch) >= self.batch_size * 2:
                    break
            if len(tactile_batch) < self.batch_size * 2:
                if dataset not in self.skipped_datasets.keys():
                    self.skipped_datasets[dataset] = int(len(tactile_batch) / 2)
                elif int(len(tactile_batch) / 2) > self.skipped_datasets[dataset]:
                    self.skipped_datasets[dataset] = int(len(tactile_batch) / 2)
                continue
            else:
                self.all_tactile += tactile_batch
                self.all_datasets += [dataset] * len(tactile_batch)
        print("Tactile-tactile contrastive all processed datasets:", set(self.datasets))
        print("Tactile-tactile contrastive skipped datasets:", self.skipped_datasets)
    
    def __len__(self): 
        return len(self.all_tactile)

    def __getitem__(self, index):
        # transforms_list = [
        #     transforms.ToTensor(),
        #     transforms.Resize(self.frame_size, interpolation=3),
        #     transforms.Normalize(
        #         mean=[0.48145466, 0.4578275, 0.40821073],
        #         std=[0.26862954, 0.26130258, 0.27577711]
        #     )
        # ]
        # if self.split_name == "train":
        #     if random.random() < self.flip_p:
        #         transforms_list.append(transforms.RandomHorizontalFlip(1))
        #     if random.random() < self.flip_p:
        #         transforms_list.append(transforms.RandomVerticalFlip(1))
        # transforms_list.append(transforms.CenterCrop(self.frame_size))
        # image_transforms = transforms.Compose(transforms_list)
        mean, std = get_dataset_img_norm(dataset)
        transforms_list = [transforms.Resize(self.frame_size, interpolation=3)]
        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        else:
            transforms_list.append(transforms.CenterCrop(self.frame_size))
        image_transforms = transforms.Compose(transforms_list)
        tactile = self.all_tactile[index]
        if self.split_name == "train":
            tactile_frames, _ = get_frames(tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        else:
            tactile_frames, _ = get_frames(tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        max_frame_length = 10
        frame_len = tactile_frames.shape[0]
        if frame_len < max_frame_length:
            new_tactile = torch.stack([tactile_frames[0]] * (max_frame_length - frame_len), dim=0)
            tactile_frames = torch.cat([new_tactile, tactile_frames], dim=0)
        return tactile_frames, self.all_datasets[index]


class TactileLLMDataset(Dataset):
    def __init__(self, image_processor, files, split_name, tokenizer, frame_size, flip_p, model_type, rag=False, tactile_vificlip=None, saved_embeddings=None, sample_tactile_paths=None, object_ids=None, device=None, retrieval_object_num=1):
        super().__init__()
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.eos_token_number = self.tokenizer.encode(self.eos_token)
        self.frame_size = frame_size
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.samples = None
        self.model_type = model_type
        self.rag = rag
        if self.rag:
            self.tactile_vificlip = tactile_vificlip
            self.saved_embeddings = saved_embeddings
            self.sample_tactile_paths = sample_tactile_paths
            self.object_ids = object_ids
            self.device = device
            self.retrieval_object_num = retrieval_object_num
        if "llama-3" in self.model_type:
            self.eot_token = "<|eot_id|>"
        for f in files:
            with open(f) as json_file:
                if self.samples is None:
                    self.samples = json.load(json_file)
                else:
                    self.samples += json.load(json_file)
                json_file.close()
    
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):
        # 1) Get sample question, answer and tactile path(s)
        sample = self.samples[index]
        tactile = sample["info"]["tactile"]
        all_objects_dict = sample["info"]["objects"]
        rag_outputs = []
        # 3) Get frame tensors
        tactile_frames = []
        all_indices = []
        all_datasets = []
        for t in tactile:
            dataset = t.split("/")[-2].split("_")[0]
            image_transforms = get_image_transforms(self.frame_size, dataset, split_name=self.split_name, flip_p=self.flip_p)
            if self.split_name == "train":
                frames, indices = get_frames(t, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
            else:
                frames, indices = get_frames(t, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
                if self.rag:
                    obj_name_description_map = get_rag_tactile_paths(frames, self.tactile_vificlip, self.saved_embeddings, self.sample_tactile_paths, self.object_ids, self.device, retrieval_object_num=self.retrieval_object_num)
                    rag_outputs.append(obj_name_description_map)
            tactile_frames.append(frames)
            all_indices.append(indices)
            all_datasets.append(dataset)
        if "scenario" in sample["info"].keys():
            # Scenario reasoning
            scenario = sample["info"]["scenario"]
            scenario_steps = len(sample["chat"])
            # if "stick" in sample["chat"][2]["content"]:
            #     target = "stick"
            # elif "wet" in sample["chat"][2]["content"]:
            #     target = "wet"
            # elif "break" in sample["chat"][2]["content"]:
            #     target = "break"
            # num_candidates = -1
            target = sample["info"]["target"]
            num_candidates = sample["info"]["num_candidates"]
            return sample["chat"], tactile_frames, tactile, all_datasets, all_indices, all_objects_dict, scenario, scenario_steps, target, num_candidates, rag_outputs
        else:
            # Descriptions and/or rankings
            question = self.tokenizer.apply_chat_template(sample["chat"][:-1], tokenize=False, add_generation_prompt=True)
            answer = sample["chat"][-1]["content"]
            if "llama-3" in self.model_type:
                answer_tokens = encode_text(self.tokenizer, answer + self.eot_token)
            else:
                answer_tokens = encode_text(self.tokenizer, answer + self.eos_token)
            return question, sample["chat"], answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict


# def get_rag_tactile_paths(original_tactile_frames, tactile_vificlip, saved_embeddings, sample_tactile_paths, object_ids, device, retrieval_object_num=1):
#     cos = nn.CosineSimilarity(dim=1, eps=1e-08)
#     original_tactile_frames = torch.unsqueeze(original_tactile_frames, dim=0)
#     sensors = [get_dataset_sensor_type("physiclear")] # NOTE: Only for non-dotted GelSight Mini
#     tactile_video_features, _, _, _ = tactile_vificlip(original_tactile_frames.to(device), None, None, sensors)
#     similarities = cos(saved_embeddings, tactile_video_features)
#     similarities_topk = torch.topk(similarities, k=retrieval_object_num+1) # NOTE
#     similar_objects = [object_ids[i] for i in similarities_topk.indices[1:]]
#     obj_name_description_map = {}
#     for obj in similar_objects:
#         obj_name = OBJECTS_PART_NAMES[obj]
#         obj_name_description_map[obj_name] = sorted(OPEN_SET_TEXTURES[obj])
#     return obj_name_description_map


def get_rag_tactile_paths(original_tactile_frames, tactile_vificlip, saved_embeddings, sample_tactile_paths, object_ids, device, retrieval_object_num=1, new_rag_object_names=[], new_rag_obj_name_description_map={}, new_rag_obj_id_map={}, new_rag_embeddings=[]):
    # NOTE: Both old and new RAG items
    cos = nn.CosineSimilarity(dim=1, eps=1e-08)
    original_tactile_frames = torch.unsqueeze(original_tactile_frames, dim=0)
    sensors = [get_dataset_sensor_type("physiclear")] # NOTE: Only for non-dotted GelSight Mini
    tactile_video_features, _, _, _ = tactile_vificlip(original_tactile_frames.to(device), None, None, sensors)
    if new_rag_embeddings is not None and new_rag_embeddings != []:
        new_rag_embeddings = torch.stack(new_rag_embeddings, dim=0).to(device)
        saved_embeddings = torch.cat([saved_embeddings, new_rag_embeddings], dim=0)
    similarities = cos(saved_embeddings, tactile_video_features)
    # similarities_topk = torch.topk(similarities, k=retrieval_object_num+1)
    # indices = [i for i in similarities_topk.indices[1:]]
    # NOTE: Do not skip the first one as all RAG items are relevant (we are only comparing new test objects)
    similarities_topk = torch.topk(similarities, k=retrieval_object_num)
    indices = [i for i in similarities_topk.indices]
    # NOTE: To check
    obj_name_description_map = {}
    for i in indices:
        if i < len(object_ids):
            # Old RAG items
            obj_id = object_ids[i]
            obj_name = OBJECTS_PART_NAMES[obj_id]
            obj_name_description_map[obj_name] = sorted(OPEN_SET_TEXTURES[obj_id])
        else:
            # New RAG items
            new_index = i - len(object_ids)
            obj_name = new_rag_object_names[new_index]
            obj_name_description_map[obj_name] = sorted(new_rag_obj_name_description_map[obj_name])
    return obj_name_description_map
    

def encode_text(tokenizer, text):
    # Remove start and end tokens if they exist
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
    if tokenizer.bos_token is not None:
        tokens = tokens[1:]
    return tokens


def get_dataset_sensor_type_old(dataset):
    dataset_sensor_map = {
        "feelang": "plain",
        "hardness": "dotted",
        "objectfolder": "plain",
        "physiclear": "plain",
        "physicleardotted": "dotted",
        "schaeffler": "plain",
        "schaefflerdotted": "dotted",
        "tvl": "plain"
    }
    return dataset_sensor_map[dataset]


def get_dataset_sensor_type(dataset):
    dataset_sensor_map = {
        "feelang": "gelsightmini",
        "hardness": "gelsight17",
        "objectfolder": "gelsight",
        "physiclear": "gelsightmini",
        "physicleardotted": "gelsightminidotted",
        "schaeffler": "gelsightmini",
        "schaefflerdotted": "gelsightminidotted",
        "tvl": "digit"
    }
    return dataset_sensor_map[dataset]


def get_dataset_img_norm(dataset):
    sensor = get_dataset_sensor_type(dataset)
    dataset_img_norm_map = {
        "gelsight": {
            "mean": [0.46640, 0.44838, 0.45375],
            "std": [0.08117, 0.07227, 0.085867],
        },
        "gelsightmini": {
            "mean": [0.20954, 0.37124, 0.40463],
            "std": [0.12138, 0.06335, 0.10677],
        },
        "gelsightminidotted": {
            "mean": [0.20251, 0.36450, 0.40409],
            "std": [0.12902, 0.08006, 0.11987],
        },
        "gelsight17": {
            "mean": [0.42487, 0.41575, 0.43780],
            "std": [0.06776, 0.06513, 0.06871],
        },
        "digit": {
            "mean": [0.41146, 0.42410, 0.39767],
            "std": [0.15062, 0.08714, 0.08073],
        },
    }
    return dataset_img_norm_map[sensor]["mean"], dataset_img_norm_map[sensor]["std"]


def get_image_transforms_old(frame_size, dataset, split_name, flip_p):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize(frame_size, interpolation=3),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ]
    if split_name == "train":
        if random.random() < flip_p:
            transforms_list.append(transforms.RandomHorizontalFlip(1))
        if random.random() < flip_p:
            transforms_list.append(transforms.RandomVerticalFlip(1))
    else:
        transforms_list.append(transforms.CenterCrop(frame_size))
    image_transforms = transforms.Compose(transforms_list)
    return image_transforms


def get_image_transforms(frame_size, dataset, split_name, flip_p):
    transforms_list = []
    mean, std = get_dataset_img_norm(dataset)
    if split_name == "train":
        transforms_list += [transforms.ColorJitter(.2,.2,.1,.02)]
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ]
    if split_name == "train":
        if random.random() < flip_p:
            transforms_list.append(transforms.RandomHorizontalFlip(1))
        if random.random() < flip_p:
            transforms_list.append(transforms.RandomVerticalFlip(1))
    image_transforms = transforms.Compose(transforms_list)
    return image_transforms


def get_frames_old(frames_path, image_processor, image_transforms, frame_size, train=True, return_indices=False):
    # Get relevant object(s) and their frames
    image_tensors = []
    all_obj_sample_frames = natsort.natsorted(os.path.join(frames_path, i) for i in os.listdir(frames_path))
    frame_indices = natsort.natsorted([int(i.split("/")[-1].split(".")[0]) for i in all_obj_sample_frames])
    # Process images
    image = Image.open(all_obj_sample_frames[0]).convert("RGB")
    image = image_transforms(image)
    if train:
        i, j, _, _ = transforms.RandomCrop.get_params(image, output_size=(frame_size, frame_size))
        image = crop(image, i, j, frame_size, frame_size)
    image_tensors.append(image)
    for frame in all_obj_sample_frames[1:]:
        image = Image.open(frame).convert("RGB")
        image = image_transforms(image)
        if train:
            image = crop(image, i, j, frame_size, frame_size)
        image_tensors.append(image)
    image_tensors = torch.stack(image_tensors, dim=0) # (l, c, h, w)
    if return_indices:
        frame_indices = [int(i.split("/")[-1].split(".")[0]) for i in all_obj_sample_frames]
        return image_tensors, frame_indices
    else:
        return image_tensors
    

def get_frames(frames_path, image_processor, image_transforms, frame_size, train=True, return_indices=False):
    # Get relevant object(s) and their frames
    image_tensors = []
    all_obj_sample_frames = natsort.natsorted(os.path.join(frames_path, i) for i in os.listdir(frames_path))
    frame_indices = natsort.natsorted([int(i.split("/")[-1].split(".")[0]) for i in all_obj_sample_frames])
    # Process images
    for frame in all_obj_sample_frames:
        image = Image.open(frame).convert("RGB")
        image = image_transforms(image)
        # if train:
        #     image = crop(image, i, j, frame_size, frame_size)
        image_tensors.append(image)
    image_tensors = torch.stack(image_tensors, dim=0) # (l, c, h, w)
    # Padding to ensure whole image is inside
    max_size = max(image_tensors.shape[-2:])
    resize_transforms_list = [
        transforms.Pad(((max_size - image_tensors.shape[-1])//2, (max_size - image_tensors.shape[-2])//2)),
        transforms.Resize(frame_size, interpolation=3)
    ]
    image_tensors = transforms.Compose(resize_transforms_list)(image_tensors)
    if return_indices:
        frame_indices = [int(i.split("/")[-1].split(".")[0]) for i in all_obj_sample_frames]
        return image_tensors, frame_indices
    else:
        return image_tensors