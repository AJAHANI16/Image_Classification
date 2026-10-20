import math

import pytest
import torch
from torch.nn import functional as F

from image_classification.metrics import classification_metrics, fit_temperature


def test_metrics_against_hand_calculation():
    # True [0, 0, 1, 1], predicted [0, 1, 1, 1].
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = classification_metrics(logits, labels, ["a", "b"])
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    confidence = 1 / (1 + math.exp(-2))
    assert metrics["expected_calibration_error"] == pytest.approx(confidence - 0.75)
    assert metrics["negative_log_likelihood"] == pytest.approx(
        (-3 * math.log(confidence) - math.log(1 - confidence)) / 4
    )
    assert metrics["brier_score"] == pytest.approx(
        (3 * 2 * (1 - confidence) ** 2 + 2 * confidence**2) / 4
    )


def test_missing_class_still_counts_in_macro_f1():
    metrics = classification_metrics(torch.tensor([[100.0, -100.0]]), torch.tensor([0]), ["a", "b"])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 0.5
    assert metrics["per_class"]["b"]["support"] == 0
    assert metrics["expected_calibration_error"] == 0.0


def test_temperature_reduces_overconfident_validation_nll_and_preserves_predictions():
    logits = torch.tensor([[10.0, 0.0]] * 4)
    labels = torch.tensor([0, 0, 0, 1])
    temperature = fit_temperature(logits, labels)
    assert 1 < temperature <= 20
    assert F.cross_entropy(logits / temperature, labels) < F.cross_entropy(logits, labels)
    assert torch.equal(logits.argmax(1), (logits / temperature).argmax(1))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperatures_rejected(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        classification_metrics(
            torch.zeros(2, 2), torch.tensor([0, 1]), ["a", "b"], temperature=temperature
        )


@pytest.mark.parametrize(
    "logits, labels",
    [
        (torch.zeros(0, 2), torch.tensor([], dtype=torch.long)),
        (torch.tensor([[float("nan"), 0.0]]), torch.tensor([0])),
        (torch.zeros(2, 2), torch.tensor([0, 2])),
        (torch.zeros(2, 2), torch.tensor([0.0, 1.0])),
    ],
)
def test_invalid_metric_inputs_rejected(logits, labels):
    with pytest.raises(ValueError):
        classification_metrics(logits, labels, ["a", "b"])
