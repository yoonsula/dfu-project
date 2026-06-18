from __future__ import annotations

import argparse

from cli.dataset_args import add_dataset_args
from trainers.common import add_common_args
from trainers.common import prepare_run
from trainers.segmentation import run_segmentation_training

TASK = "foot"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the foot segmentation head with a frozen DINOv3 backbone.",
    )
    add_dataset_args(parser)
    add_common_args(parser)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def train(args: argparse.Namespace) -> None:
    device, use_amp = prepare_run(args)
    run_segmentation_training(args, device, use_amp, task=TASK)


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
