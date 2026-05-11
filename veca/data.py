from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def require_data_paths(cfg) -> None:
    assert cfg.train_dir, "Set OBJECT365_TRAIN_DIR or OBJECT365_ROOT."
    assert cfg.val_dir, "Set OBJECT365_VAL_DIR or OBJECT365_ROOT."
    assert Path(cfg.train_dir).exists(), f"Training directory not found: {cfg.train_dir}"
    assert Path(cfg.val_dir).exists(), f"Validation directory not found: {cfg.val_dir}"


def ensure_index_file(root_dir: str, out_txt: str) -> None:
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        return
    root = Path(root_dir)
    assert root.exists(), f"Dataset root not found: {root_dir}"
    print(f"[Index] Building index: {out_txt}")
    n = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                f.write(str(p.resolve()) + "\n")
                n += 1
                if n % 1_000_000 == 0:
                    print(f"[Index] wrote {n:,} paths")
    assert n > 0, f"No images found under: {root_dir}"
    print(f"[Index] Done. Total images indexed: {n:,}")


def load_paths(index_file: str) -> List[str]:
    print(f"[Dataset] Loading paths from {index_file}")
    paths: List[str] = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    print(f"[Dataset] Loaded {len(paths):,} paths")
    return paths


def make_transform(image_size: int, resize_short: int, *, is_train: bool):
    to_img = v2.ToImage()
    to_dtype = v2.ToDtype(torch.float32, scale=True)
    norm = v2.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    if is_train:
        return v2.Compose([
            to_img,
            v2.Resize(resize_short, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.RandomCrop(image_size),
            v2.RandomHorizontalFlip(),
            v2.TrivialAugmentWide(),
            to_dtype,
            norm,
        ])
    return v2.Compose([
        to_img,
        v2.Resize(resize_short, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(image_size),
        to_dtype,
        norm,
    ])


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        super().__init__()
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    is_train: bool,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=is_train,
        drop_last=is_train,
    )
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        sampler=sampler,
        shuffle=False,
        drop_last=is_train,
    )
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **common)
    return loader, sampler


def create_object365_dataloaders_ddp(cfg, rank: int, world_size: int):
    train_t = make_transform(cfg.image_size, cfg.resize_short, is_train=True)
    val_t = make_transform(cfg.image_size, cfg.resize_short, is_train=False)
    train_ds = ImagePathDataset(load_paths(cfg.train_index), train_t)
    val_ds = ImagePathDataset(load_paths(cfg.val_index), val_t)
    train_loader, train_sampler = build_loader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        rank=rank,
        world_size=world_size,
        is_train=True,
    )
    val_loader, val_sampler = build_loader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        rank=rank,
        world_size=world_size,
        is_train=False,
    )
    return train_loader, val_loader, train_sampler, val_sampler


class MultiResLoaders:
    def __init__(self, loaders: Dict[int, DataLoader], samplers: Dict[int, DistributedSampler]):
        self.loaders = loaders
        self.samplers = samplers
        self.iters: Dict[int, object] = {r: iter(loader) for r, loader in loaders.items()}
        self.sampler_epochs: Dict[int, int] = {r: 0 for r in loaders.keys()}

    def set_epoch_all(self, epoch: int) -> None:
        for r, sampler in self.samplers.items():
            self.sampler_epochs[r] = epoch
            sampler.set_epoch(epoch)
            self.iters[r] = iter(self.loaders[r])

    def next_batch(self, resolution: int):
        try:
            return next(self.iters[resolution])
        except StopIteration:
            self.sampler_epochs[resolution] += 1
            self.samplers[resolution].set_epoch(self.sampler_epochs[resolution])
            self.iters[resolution] = iter(self.loaders[resolution])
            return next(self.iters[resolution])


def create_multires_object365_loaders_ddp(cfg, rank: int, world_size: int):
    train_paths = load_paths(cfg.train_index)
    val_paths = load_paths(cfg.val_index)
    train_loaders = {}
    train_samplers = {}
    val_loaders = {}
    val_samplers = {}
    for res in cfg.resolutions:
        resize_short = cfg.resize_short_by_res[res]
        train_t = make_transform(res, resize_short, is_train=True)
        val_t = make_transform(res, resize_short, is_train=False)
        train_ds = ImagePathDataset(train_paths, train_t)
        val_ds = ImagePathDataset(val_paths, val_t)
        train_loaders[res], train_samplers[res] = build_loader(
            train_ds,
            batch_size=cfg.batch_size_by_res[res],
            num_workers=cfg.train_num_workers,
            rank=rank,
            world_size=world_size,
            is_train=True,
            prefetch_factor=cfg.prefetch_factor,
        )
        val_loaders[res], val_samplers[res] = build_loader(
            val_ds,
            batch_size=cfg.batch_size_by_res[res],
            num_workers=cfg.val_num_workers,
            rank=rank,
            world_size=world_size,
            is_train=False,
            prefetch_factor=cfg.prefetch_factor,
        )
    return MultiResLoaders(train_loaders, train_samplers), MultiResLoaders(val_loaders, val_samplers)
