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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from typing import TypeVar, Optional, Iterator
T_co = TypeVar('T_co', covariant=True)


def get_frames(frames_path, image_processor, image_transforms, frame_size, train=True, return_indices=False):
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


class DistributedSampler(DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # TODO: Change this so each batch still has all unique objects from only one dataset when shuffled
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    

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
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
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
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        transforms_list.append(transforms.CenterCrop(self.frame_size))
        image_transforms = transforms.Compose(transforms_list)
        # tactile_anchor = self.tactile[dataset][index]
        # all_tactile_frames = []
        # if self.split_name == "train":
        #     tactile_frames, _ = get_frames(tactile_anchor, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        # else:
        #     tactile_frames, _ = get_frames(tactile_anchor, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        # all_tactile_frames.append(tactile_frames) # [(l, c, h, w)]
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
        # similar_tactile = random.choice(self.tactile_by_object_id[dataset][current_object_id])
        # while tactile_anchor == similar_tactile:
        #     similar_tactile = random.choice(self.tactile_by_object_id[dataset][current_object_id])
        #     # print(dataset, index, tactile_anchor, similar_tactile)
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
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
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
    

class TactileTextContrastiveDataset(Dataset):
    def __init__(self, image_processor, tokenizer, data_path, split_name, datasets, frame_size, flip_p=0, batch_size=None):
        super().__init__()
        self.split_name = split_name
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tactile = {}
        self.object_description = {}
        self.tactile_by_object_description = {}
        self.datasets = []
        self.frame_size = frame_size
        self.num_samples = 0
        for sample in os.listdir(data_path):
            sample_dataset = sample.split("_")[0]
            try:
                data = json.load(open(os.path.join(data_path, sample + "/data.json"), "r"))
            except FileNotFoundError:
                continue
            if split_name != data['split'] or "tactile" not in os.listdir(os.path.join(data_path, sample)) or sample_dataset not in datasets or "object" not in data.keys():
                continue
            if sample_dataset not in self.datasets:
                self.datasets.append(sample_dataset)
                self.tactile[sample_dataset] = []
                self.object_description[sample_dataset] = []
                self.tactile_by_object_description[sample_dataset] = {}
            self.num_samples += 1
            # Save object description depending on dataset
            description = f"A tactile sensor video of {data['object']}."
            self.object_description[sample_dataset].append(description)
            # Save tactile depending on dataset and object description
            if data["object"] not in self.tactile_by_object_description[sample_dataset].keys():
                self.tactile_by_object_description[sample_dataset][data["object"]] = [os.path.join(data_path, sample + "/tactile")]
            else:
                self.tactile_by_object_description[sample_dataset][data["object"]].append(os.path.join(data_path, sample + "/tactile"))
            # Save tactile depending on dataset
            self.tactile[sample_dataset].append(os.path.join(data_path, sample + "/tactile"))
    
    def __len__(self): 
        return self.num_samples

    def __getitem__(self, index):
        # 1) choose dataset
        dataset = random.choice(self.datasets)
        index = index % len(self.tactile[dataset])
        # 2) Get tactile data and similar object description
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        transforms_list.append(transforms.CenterCrop(self.frame_size))
        image_transforms = transforms.Compose(transforms_list)
        tactile_anchor = self.tactile[dataset][index]
        all_tactile_frames = []
        all_descriptions = []
        if self.split_name == "train":
            tactile_frames, _ = get_frames(tactile_anchor, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
        else:
            tactile_frames, _ = get_frames(tactile_anchor, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
        all_tactile_frames.append(tactile_frames) # [(l, c, h, w)]
        current_object_description = self.object_description[dataset][index]
        description_ids = self.tokenizer(current_object_description, padding=True, return_tensors="pt")
        description_ids = description_ids["input_ids"][0]
        all_descriptions.append(description_ids)
        # 3) Get dissimilar object descriptions
        other_tactile_frames = []
        other_descriptions = []
        object_descriptions = list(self.tactile_by_object_description[dataset].keys())
        random.shuffle(object_descriptions)
        for object_description in object_descriptions:
            if object_description != current_object_description:
                other_tactile = random.choice(self.tactile_by_object_description[dataset][object_description])
                if self.split_name == "train":
                    tactile_frames, _ = get_frames(other_tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
                else:
                    tactile_frames, _ = get_frames(other_tactile, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
                other_tactile_frames.append(tactile_frames)
                description_ids = self.tokenizer(object_description, padding=True, return_tensors="pt")
                description_ids = description_ids["input_ids"][0]
                other_descriptions.append(description_ids)
            if len(other_tactile_frames) >= self.batch_size - 1:
                break
        all_tactile_frames += other_tactile_frames
        all_descriptions += other_descriptions
        num_objects = len(all_tactile_frames)
        max_frame_length = 0
        for i in range(num_objects):
            frame_len = all_tactile_frames[i].shape[0]
            if frame_len > max_frame_length:
                max_frame_length = frame_len
        for i in range(num_objects):
            # Tactile
            tactile = all_tactile_frames[i]
            frame_len = tactile.shape[0]
            if frame_len < max_frame_length:
                new_tactile = torch.stack([all_tactile_frames[i][0]] * (max_frame_length - frame_len), dim=0)
                all_tactile_frames[i] = torch.cat([new_tactile, all_tactile_frames[i]], dim=0)
        all_tactile_frames = torch.stack(all_tactile_frames)
        all_descriptions = pad_sequence(all_descriptions, batch_first=True)
        return all_tactile_frames, all_descriptions, dataset
    

class TactileTextContrastiveDistributedDataset(Dataset):
    def __init__(self, image_processor, tokenizer, data_path, split_name, datasets, frame_size, flip_p=0, batch_size=None):
        super().__init__()
        self.split_name = split_name
        self.flip_p = flip_p
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.object_description = {}
        self.tactile = {}
        self.tactile_by_object_description = {}
        self.object_description_ids = {}
        self.datasets = []
        self.frame_size = frame_size
        self.num_samples = 0
        self.all_tactile_text = []
        self.all_datasets = []
        self.skipped_datasets = {}
        for sample in os.listdir(data_path):
            sample_dataset = sample.split("_")[0]
            try:
                data = json.load(open(os.path.join(data_path, sample + "/data.json"), "r"))
            except FileNotFoundError:
                continue
            if split_name != data['split'] or "tactile" not in os.listdir(os.path.join(data_path, sample)) or sample_dataset not in datasets or "object" not in data.keys():
                continue
            if sample_dataset not in self.datasets:
                self.datasets.append(sample_dataset)
                self.object_description[sample_dataset] = []
                self.tactile[sample_dataset] = []
                self.tactile_by_object_description[sample_dataset] = {}
                self.object_description_ids[sample_dataset] = []
            self.num_samples += 1
            # Save object description depending on dataset
            description = f"A tactile sensor video of {data['object']}."
            self.object_description[sample_dataset].append(description)
            description_id = self.tokenizer(description, padding=True, return_tensors="pt")
            description_id = description_id["input_ids"][0]
            # Save object description tensor depending on dataset
            self.object_description_ids[sample_dataset].append(description_id)
            # Save tactile depending on dataset and object description
            if data["object"] not in self.tactile_by_object_description[sample_dataset].keys():
                self.tactile_by_object_description[sample_dataset][description] = [os.path.join(data_path, sample + "/tactile")]
            else:
                self.tactile_by_object_description[sample_dataset][description].append(os.path.join(data_path, sample + "/tactile"))
            # Save tactile depending on dataset
            self.tactile[sample_dataset].append(os.path.join(data_path, sample + "/tactile"))
        for k, v in self.object_description_ids.items():
            self.object_description_ids[k] = pad_sequence(v, batch_first=True)

        # Fill up samples based on batch size
        num_batches = 100
        for i in range(num_batches):
            # 1) Choose dataset
            dataset = random.choice(self.datasets)
            index = random.randint(0, len(self.object_description[dataset])-1)
            valid_current_object_description = False
            current_object_description = self.object_description[dataset][index]
            while not valid_current_object_description:
                # 2) Get tactile data
                try:
                    tactile_set = random.sample(self.tactile_by_object_description[dataset][current_object_description], 1)
                    valid_current_object_description = True
                except ValueError:
                    index += 1
                    current_object_description = self.object_description[dataset][index]
            tactile_batch = [tactile_set[0], self.object_description_ids[dataset][index]]
            # 3) Get dissimilar tactile data
            object_descriptions = list(self.tactile_by_object_description[dataset].keys())
            random.shuffle(object_descriptions)
            for object_description in object_descriptions:
                if object_description != current_object_description:
                    try:
                        tactile_set = random.sample(self.tactile_by_object_description[dataset][object_description], 1)
                    except ValueError:
                        continue
                    tactile_batch.append(tactile_set[0])
                    # Pad description if necessary
                    max_description_length = self.object_description_ids[dataset].shape[-1]
                    description_id = self.tokenizer(object_description, padding=True, return_tensors="pt")
                    description_id = description_id["input_ids"][0]
                    if len(description_id) < max_description_length:
                        len_diff = max_description_length - len(description_id)
                        description_id = torch.cat([description_id, torch.tensor([0] * len_diff)], dim=-1)
                    tactile_batch.append(description_id)
                if len(tactile_batch) >= self.batch_size * 2:
                    break
            if len(tactile_batch) < self.batch_size * 2:
                if dataset not in self.skipped_datasets.keys():
                    self.skipped_datasets[dataset] = int(len(tactile_batch) / 2)
                elif int(len(tactile_batch) / 2) > self.skipped_datasets[dataset]:
                    self.skipped_datasets[dataset] = int(len(tactile_batch) / 2)
                continue
            else:
                self.all_tactile_text += tactile_batch
                self.all_datasets += [dataset] * len(tactile_batch)
        print("Tactile-text contrastive all processed datasets:", set(self.datasets))
        print("Tactile-text contrastive skipped datasets:", self.skipped_datasets)
    
    def __len__(self): 
        return len(self.all_tactile_text)

    def __getitem__(self, index):
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        transforms_list.append(transforms.CenterCrop(self.frame_size))
        image_transforms = transforms.Compose(transforms_list)
        tactile_or_text = self.all_tactile_text[index]
        if type(tactile_or_text) == str:
            # Tactile
            if self.split_name == "train":
                tactile_frames, _ = get_frames(tactile_or_text, self.image_processor, image_transforms, frame_size=self.frame_size, train=True, return_indices=True)
            else:
                tactile_frames, _ = get_frames(tactile_or_text, self.image_processor, image_transforms, frame_size=self.frame_size, train=False, return_indices=True)
            max_frame_length = 10
            frame_len = tactile_frames.shape[0]
            if frame_len < max_frame_length:
                new_tactile = torch.stack([tactile_frames[0]] * (max_frame_length - frame_len), dim=0)
                tactile_frames = torch.cat([new_tactile, tactile_frames], dim=0)
            return tactile_frames, "tactile", self.all_datasets[index]
        else:
            # Text
            return tactile_or_text, "text", self.all_datasets[index]


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
        # NOTE: ignore BOS tokens
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize(self.frame_size, interpolation=3),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transforms_list.append(transforms.RandomVerticalFlip(1))
        else:
            transforms_list.append(transforms.CenterCrop(self.frame_size))
        transforms_image = transforms.Compose(transforms_list)
        sample = self.samples[index]
        tactile = sample["info"]["tactile"]
        all_objects_dict = sample["info"]["objects"]
        rag_outputs = []
        # 3) Get frame tensors
        tactile_frames = []
        all_indices = []
        all_datasets = []
        for t in tactile:
            if self.split_name == "train":
                frames, indices = get_frames(t, self.image_processor, transforms_image, frame_size=self.frame_size, train=True, return_indices=True)
            else:
                frames, indices = get_frames(t, self.image_processor, transforms_image, frame_size=self.frame_size, train=False, return_indices=True)
                if self.rag:
                    obj_name_description_map = get_rag_tactile_paths(frames, self.tactile_vificlip, self.saved_embeddings, self.sample_tactile_paths, self.object_ids, self.device, retrieval_object_num=self.retrieval_object_num)
                    rag_outputs.append(obj_name_description_map)
            tactile_frames.append(frames)
            all_indices.append(indices)
            dataset = t.split("/")[-2].split("_")[0]
            all_datasets.append(dataset)
        if "scenario" in sample["info"].keys():
            # Scenario reasoning
            scenario = sample["info"]["scenario"]
            scenario_steps = len(sample["chat"])
            target = sample["info"]["target"]
            num_candidates = sample["info"]["num_candidates"]
            return sample["chat"], tactile_frames, tactile, all_datasets, all_indices, all_objects_dict, scenario, scenario_steps, target, num_candidates, rag_outputs
        else:
            # Descriptions and/or rankings
            question = self.tokenizer.apply_chat_template(sample["chat"][:-1], tokenize=False, add_generation_prompt=True)
            answer = sample["chat"][-1]["content"] # NOTE: Original
            if "llama-3" in self.model_type:
                answer_tokens = encode_text(self.tokenizer, answer + self.eot_token)
            else:
                answer_tokens = encode_text(self.tokenizer, answer + self.eos_token)
            return question, sample["chat"], answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict
        

def get_rag_tactile_paths(original_tactile_frames, tactile_vificlip, saved_embeddings, sample_tactile_paths, object_ids, device, retrieval_object_num=1):
    cos = nn.CosineSimilarity(dim=1, eps=1e-08)
    original_tactile_frames = torch.unsqueeze(original_tactile_frames, dim=0)
    tactile_video_features, _, _, _ = tactile_vificlip(original_tactile_frames.to(device), None, None)
    similarities = cos(saved_embeddings, tactile_video_features)
    similarities_topk = torch.topk(similarities, k=retrieval_object_num)
    similar_objects = [object_ids[i] for i in similarities_topk.indices]
    obj_name_description_map = {}
    for obj in similar_objects:
        obj_name = OBJECTS_PART_NAMES[obj]
        obj_name_description_map[obj_name] = sorted(OPEN_SET_TEXTURES[obj])
    return obj_name_description_map
    

def encode_text(tokenizer, text):
    # Remove start and end tokens if they exist
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
    if tokenizer.bos_token is not None:
        tokens = tokens[1:]
    return tokens


def get_dataset_sensor_type(dataset):
    if dataset == "physiclear" or dataset == "objectfolder":
        return "plain"
    elif dataset == "physicleardotted" or dataset == "hardness":
        return "dotted"