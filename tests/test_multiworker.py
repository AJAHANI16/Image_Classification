import pytest

from image_classification.engine import TrainConfig, train


@pytest.mark.multiworker
def test_spawned_workers_complete_training(dataset_root, tmp_path):
    report = train(
        TrainConfig(
            data=dataset_root,
            output=tmp_path / "run",
            model="tiny_cnn",
            pretrained=False,
            epochs=2,
            warmup_epochs=0,
            image_size=32,
            batch_size=4,
            device="cpu",
            workers=2,
        )
    )
    assert report["test"]["samples"] == 4
