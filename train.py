from __future__ import annotations

import argparse
import sys

from trainers.dfu_trainer import main as dfu_main
from trainers.foot_trainer import main as foot_main
from trainers.wound_trainer import main as wound_main

TRAINERS = {
    "foot": foot_main,
    "wound": wound_main,
    "dfu": dfu_main,
}


def parse_dispatch_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train one DFU task head with a frozen DINOv3 backbone.",
        add_help=False,
    )
    parser.add_argument("--task", type=str, choices=tuple(TRAINERS), required=True)
    args, remaining = parser.parse_known_args(argv)
    return args.task, remaining


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (len(argv) == 1 and argv[0] in {"-h", "--help"}):
        print(
            "Usage: python train.py --task {foot,wound,dfu} [trainer options]\n\n"
            "Task-specific entry points:\n"
            "  python -m trainers.foot_trainer\n"
            "  python -m trainers.wound_trainer\n"
            "  python -m trainers.dfu_trainer"
        )
        return

    if "--help" in argv or "-h" in argv:
        if "--task" in argv:
            task_index = argv.index("--task")
            if task_index + 1 < len(argv):
                task = argv[task_index + 1]
                if task in TRAINERS:
                    trainer_argv = [
                        arg
                        for index, arg in enumerate(argv)
                        if index not in {task_index, task_index + 1}
                    ]
                    TRAINERS[task](trainer_argv)
                    return

    task, remaining = parse_dispatch_args(argv)
    TRAINERS[task](remaining)


if __name__ == "__main__":
    main(sys.argv[1:])
