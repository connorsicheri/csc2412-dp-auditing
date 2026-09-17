"""Data loading utilities for MNIST and CIFAR-10."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _get_transforms(dataset_name: str, train: bool = False) -> transforms.Compose:
    """Return transforms for the given dataset.

    Parameters
    ----------
    dataset_name : str
        'mnist' or 'cifar10'.
    train : bool
        If True, include data augmentation (random crop, horizontal flip).
        Standard for DP-SGD training on CIFAR-10 (De et al. 2022).
    """
    if dataset_name == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif dataset_name == "cifar10":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_num_classes(dataset_name: str) -> int:
    """Return the number of classes for the dataset."""
    return {"mnist": 10, "cifar10": 10}[dataset_name]


def get_input_shape(dataset_name: str) -> tuple[int, ...]:
    """Return (C, H, W) for the dataset."""
    return {"mnist": (1, 28, 28), "cifar10": (3, 32, 32)}[dataset_name]


def load_dataset(
    cfg: dict[str, Any],
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Load train and test datasets based on config.

    Returns
    -------
    train_dataset, test_dataset
    """
    name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "./data")
    train_transform = _get_transforms(name, train=True)
    test_transform = _get_transforms(name, train=False)

    if name == "mnist":
        train = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
    elif name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train, test


def build_dataloaders(
    train_dataset: datasets.VisionDataset,
    test_dataset: datasets.VisionDataset,
    batch_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Split train into train/val and return dataloaders.

    Returns
    -------
    dict with keys "train", "val", "test"
    """
    n = len(train_dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)

    if n_val > 0:
        train_sub, val_sub = random_split(train_dataset, [n_train, n_val], generator=generator)
    else:
        train_sub = train_dataset
        val_sub = None

    loaders = {
        "train": DataLoader(
            train_sub, batch_size=min(batch_size, len(train_sub)), shuffle=True, drop_last=True
        ),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }
    if val_sub is not None:
        loaders["val"] = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    else:
        loaders["val"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return loaders
