import math

import pytest

from image_classification.engine import TrainConfig, evaluate, predict, train


@pytest.mark.pretrained
@pytest.mark.parametrize("architecture", ["efficientnet_v2_s", "convnext_tiny"])
def test_pretrained_two_stage_training_and_checkpoint_reload(architecture, dataset_root, tmp_path):
    output = tmp_path / architecture
    report = train(
        TrainConfig(
            data=dataset_root,
            output=output,
            model=architecture,
            pretrained=True,
            image_size=64,
            epochs=2,
            warmup_epochs=1,
            batch_size=8,
            device="cpu",
            workers=0,
            balance_classes=True,
        )
    )
    assert math.isfinite(report["test"]["negative_log_likelihood"])
    evaluation = evaluate(output / "best.pt", dataset_root, device="cpu")
    assert evaluation["raw"]["accuracy"] == report["test"]["accuracy"]
    prediction = predict(output / "best.pt", dataset_root / "red" / "000.png", device="cpu")
    assert prediction["predictions"][0]["predicted_class"] in {"red", "blue"}
    assert math.isfinite(prediction["predictions"][0]["top_k"][0]["probability"])
