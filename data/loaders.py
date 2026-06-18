from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from cli.dataset_args import foot_roots_for_args
from datasets import DiabeticFootDataset


def make_loader(
    task: str,
    split: str,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = DiabeticFootDataset(
        task=task,
        split=split,
        foot_roots=foot_roots_for_args(args),
        body_root=args.body_root,
        humanbody_root=args.humanbody_root,
        wound_root=args.wound_root,
        wound_image_root=None if args.no_wound_image else args.wound_image_root,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        val_negative_ratio=args.val_negative_ratio if task == "foot" else 0.0,
        seed=args.seed,
        negative_oversample=args.negative_oversample if task == "foot" else 1,
        neg_sample_weight=args.neg_loss_weight if task == "foot" else 1.0,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )
