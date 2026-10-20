"""Command-line entry point; progress goes to stderr and results are JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from .engine import TrainConfig, evaluate, predict, save_json, train
from .models import ARCHITECTURES
from .smoke import smoke_test


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, evaluate, and use image classifiers")
    parser.add_argument("--threads", type=int, help="CPU intra-op threads (use 2 for small jobs)")
    subcommands = parser.add_subparsers(dest="command", required=True)
    training = subcommands.add_parser(
        "train", help="Train and evaluate using audited image folders"
    )
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True, help="A new or empty run directory")
    training.add_argument("--model", choices=ARCHITECTURES, default="efficientnet_v2_s")
    training.add_argument("--epochs", type=int, default=12)
    training.add_argument("--warmup-epochs", type=int, default=None, help="Frozen-backbone epochs")
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument(
        "--image-size", type=int, help="Defaults to the selected weights' crop size"
    )
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--label-smoothing", type=float, default=0.1)
    training.add_argument("--val-fraction", type=float, default=0.15)
    training.add_argument("--test-fraction", type=float, default=0.15)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    for name, default, help_text in (
        ("pretrained", True, "Download/use ImageNet initialization"),
        ("amp", True, "Automatic mixed precision on CUDA"),
        ("augment", True, "Random crops, RandAugment, and optional horizontal flips"),
        ("hflip", True, "Horizontal flips in the training augmentation pipeline"),
        ("mixup", True, "Randomly apply MixUp or CutMix to training batches"),
        ("balance-classes", False, "Use inverse-frequency sampling on training data"),
        ("calibrate", True, "Fit temperature using validation data after model selection"),
    ):
        training.add_argument(
            f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text
        )

    evaluation = subcommands.add_parser("evaluate", help="Evaluate a saved, verified data split")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True, help="Original dataset root")
    evaluation.add_argument(
        "--manifest", type=Path, help="Defaults to splits.json beside checkpoint"
    )
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    evaluation.add_argument("--output", type=Path, help="Optional JSON report path")

    prediction = subcommands.add_parser("predict", help="Predict one image or a folder of images")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True, help="Image or directory")
    prediction.add_argument("--top-k", type=int, default=3)
    prediction.add_argument("--batch-size", type=int, default=16)
    prediction.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    prediction.add_argument("--output", type=Path, help="Optional JSON predictions path")

    smoke = subcommands.add_parser("smoke", help="Offline synthetic train/evaluate/predict check")
    smoke.add_argument("--output", type=Path, default=Path("runs/smoke"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.threads is not None:
            if args.threads < 1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
        if args.command == "train":
            options = vars(args).copy()
            options.pop("command")
            options.pop("threads")
            options["pretrained"] = args.pretrained and args.model != "tiny_cnn"
            if args.warmup_epochs is None:
                options["warmup_epochs"] = (
                    min(2, max(0, args.epochs - 1)) if (options["pretrained"]) else 0
                )
            result = train(TrainConfig(**options))
        elif args.command == "evaluate":
            result = evaluate(
                args.checkpoint,
                args.data,
                split=args.split,
                manifest_path=args.manifest,
                device=args.device,
                batch_size=args.batch_size,
            )
            if args.output:
                save_json(args.output, result)
        elif args.command == "predict":
            result = predict(
                args.checkpoint,
                args.image,
                top_k=args.top_k,
                device=args.device,
                batch_size=args.batch_size,
            )
            if args.output:
                save_json(args.output, result)
        else:
            result = smoke_test(args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
