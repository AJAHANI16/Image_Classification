import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from image_classification.data import Manifest
from image_classification.engine import TrainConfig, evaluate, load_checkpoint, predict, train


@pytest.fixture
def trained_run(dataset_root, tmp_path):
    output = tmp_path / "run"
    config = TrainConfig(
        data=dataset_root,
        output=output,
        model="tiny_cnn",
        pretrained=False,
        epochs=3,
        warmup_epochs=1,
        image_size=32,
        batch_size=5,
        learning_rate=0.01,
        backbone_lr_ratio=1.0,
        balance_classes=True,
        augment=True,
        mixup=True,
    )
    report = train(config)
    return config, report


def test_augmented_training_checkpoint_reload_evaluation_and_batch_prediction(trained_run):
    config, report = trained_run
    assert {path.name for path in config.output.iterdir()} == {
        "best.pt",
        "config.json",
        "splits.json",
        "metrics.json",
        "history.jsonl",
    }
    history = [
        json.loads(line) for line in (config.output / "history.jsonl").read_text().splitlines()
    ]
    assert history[0]["stage"] == "head"
    assert history[1]["stage"] == "fine_tune"
    assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in history)
    checkpoint = config.output / "best.pt"
    evaluation = evaluate(checkpoint, config.data, device="cpu", batch_size=3)
    assert evaluation["raw"]["confusion_matrix"] == report["test"]["confusion_matrix"]
    assert evaluation["calibrated"]["negative_log_likelihood"] == pytest.approx(
        report["test_calibrated"]["negative_log_likelihood"], abs=1e-6
    )
    predictions = predict(checkpoint, config.data, device="cpu", batch_size=7, top_k=10)
    assert len(predictions["predictions"]) == 24
    assert all(len(row["top_k"]) == 2 for row in predictions["predictions"])
    assert all(
        sum(item["probability"] for item in row["top_k"]) == pytest.approx(1.0)
        for row in predictions["predictions"]
    )
    model, payload, spec = load_checkpoint(checkpoint, torch.device("cpu"))
    assert spec.image_size == 32
    assert payload["classes"] == ["blue", "red"]
    assert payload["temperature"] > 0
    assert not model.training
    assert report["validation_calibrated"]["negative_log_likelihood"] <= (
        report["validation"]["negative_log_likelihood"] + 1e-6
    )


def test_evaluation_rejects_tampered_manifest(trained_run):
    config, _ = trained_run
    path = config.output / "splits.json"
    manifest = Manifest.load(path)
    manifest.classes.reverse()
    manifest.save(path)
    with pytest.raises(ValueError, match="does not match"):
        evaluate(config.output / "best.pt", config.data, device="cpu")


def test_existing_run_cannot_be_overwritten(trained_run):
    config, _ = trained_run
    before = (config.output / "best.pt").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        train(config)
    assert (config.output / "best.pt").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": 0},
        {"warmup_epochs": 3},
        {"batch_size": 0},
        {"workers": -1},
        {"label_smoothing": 1.0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"seed": -1},
    ],
)
def test_bad_training_options_are_rejected_before_creating_output(dataset_root, tmp_path, change):
    config = TrainConfig(data=dataset_root, output=tmp_path / "unused", epochs=3)
    with pytest.raises(ValueError):
        train(replace(config, **change))
    assert not config.output.exists()


def test_cli_offline_smoke_then_separate_prediction_process(tmp_path):
    output = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, "-m", "image_classification", "smoke", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["synthetic_test_accuracy"] == 1.0
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "image_classification",
            "--threads",
            "2",
            "predict",
            "--checkpoint",
            str(output / "run" / "best.pt"),
            "--image",
            str(output / "data" / "blue" / "000.png"),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout)["predictions"][0]["predicted_class"] == "blue"


def test_cli_reports_invalid_arguments_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "image_classification",
            "train",
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "run"),
            "--epochs",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
