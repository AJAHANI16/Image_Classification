"""A small offline integration check. Synthetic accuracy is not a real-data benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .engine import TrainConfig, evaluate, predict, save_json, train


def create_demo_data(root: Path, *, samples_per_class: int = 12) -> None:
    rng = np.random.default_rng(42)
    for name, channel in (("blue", 2), ("red", 0)):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            pixels = rng.integers(0, 35, size=(40, 48, 3), dtype=np.uint8)
            pixels[:, :, channel] += 190
            Image.fromarray(pixels).save(directory / f"{index:03}.png")


def smoke_test(output: Path) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Smoke output is not empty; choose a new directory")
    torch.set_num_threads(2)
    data, run = output / "data", output / "run"
    create_demo_data(data)
    report = train(
        TrainConfig(
            data=data,
            output=run,
            model="tiny_cnn",
            epochs=4,
            warmup_epochs=0,
            batch_size=8,
            image_size=32,
            learning_rate=0.01,
            backbone_lr_ratio=1.0,
            label_smoothing=0.0,
            device="cpu",
            pretrained=False,
            augment=False,
            mixup=False,
        )
    )
    evaluated = evaluate(run / "best.pt", data, device="cpu")
    predicted = predict(run / "best.pt", data / "red" / "000.png", device="cpu")
    if report["test"]["accuracy"] < 0.99:
        raise RuntimeError("Smoke classifier did not learn the simple synthetic task")
    if abs(report["test"]["accuracy"] - evaluated["raw"]["accuracy"]) > 1e-6:
        raise RuntimeError("Reloaded checkpoint evaluation does not match training")
    if predicted["predictions"][0]["predicted_class"] != "red":
        raise RuntimeError("Reloaded checkpoint prediction failed")
    summary = {
        "status": "passed",
        "synthetic_test_accuracy": report["test"]["accuracy"],
        "checkpoint": str(run / "best.pt"),
        "prediction": predicted["predictions"][0],
        "note": "Synthetic workflow check only; this does not measure real-world model quality.",
    }
    save_json(output / "smoke_result.json", summary)
    return summary
