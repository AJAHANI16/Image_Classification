# Image Classification Lab

A complete PyTorch project for training **single-label, multi-class image classifiers** on your own images. Includes pretrained EfficientNetV2-S and ConvNeXt-Tiny, reproducible data splits, two-stage fine-tuning, confidence calibration, and JSON evaluation reports.

This repository started empty. No real dataset or trained production model is included, and no real-world accuracy claim is made. The offline demo checks the software on an intentionally simple synthetic task.

## Quick start

Python 3.10+ is required. Create a separate environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
image-classification smoke --output runs/smoke
```

The smoke command creates synthetic red/blue images, trains a small CNN on CPU, calibrates its confidence, reloads the checkpoint, and verifies both evaluation and prediction. It does not download data or weights.

For an NVIDIA GPU, install the matching **current PyTorch/CUDA build** using the [official installation selector](https://pytorch.org/get-started/locally/) before installing this project. For CPU-only Linux, you can first run:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The default device order is CUDA, then Apple MPS, then CPU. Intel Macs use CPU. Mixed precision is enabled only on CUDA. GPU and MPS paths require validation on your hardware; the local verification described below uses CPU.

## Your dataset

Place one class in each folder, for example:

```text
data/pets/cat/image_001.jpg
data/pets/cat/image_002.jpg
data/pets/dog/image_001.jpg
data/pets/dog/image_002.jpg
```

Use **at least three unique images per class** so all three splits can contain every class; meaningful experiments need many more. The default split is approximately 70% train, 15% validation, and 15% test, stratified by class. Small classes reserve at least one image for each held-out split. Class names are sorted and saved with the checkpoint.

Already have separate splits? Supply `data/pets/train/<class>/...`, `data/pets/val/<class>/...`, and `data/pets/test/<class>/...`. All three must contain the same classes. Existing partitions are preserved, and fractional split arguments apply only to unsplit data. The names `train`, `val`, and `test` are reserved at the dataset root.

Supported formats: JPEG, PNG, BMP, WebP, TIFF. Images are EXIF-oriented and converted to RGB. Unreadable images, empty classes, conflicting labels, and duplicate decoded RGB images are rejected before training. A content-hashed manifest records the exact splits, and later evaluation rejects changed files or mismatched manifests.

Exact duplicate checks cannot detect every near-duplicate or correlated image. For multiple images of the same person, object, video, location, or session, **create your own group-separated train/val/test folders** before training. Random image-level splits are unsuitable when groups overlap.

## Train

```bash
image-classification train \
  --data data/pets \
  --output runs/pets-efficientnet \
  --model efficientnet_v2_s \
  --epochs 15 \
  --batch-size 16
```

For another strong backbone:

```bash
image-classification train \
  --data data/pets \
  --output runs/pets-convnext \
  --model convnext_tiny \
  --epochs 15 \
  --balance-classes
```

Pretrained models download official ImageNet weights on the first training run. Inference and checkpoint evaluation never download weights. `--no-pretrained` trains from scratch; frozen warmup then defaults to zero. `tiny_cnn` is an offline debugging model, not the recommended model for real tasks.

| Technique | Behavior |
| --- | --- |
| Transfer learning | EfficientNetV2-S or ConvNeXt-Tiny with a new classification head |
| Two-stage fine-tuning | Default: train the head for up to two epochs, then unfreeze the backbone |
| Differential learning rates | Backbone learns at 0.1× the head rate by default |
| Optimization | AdamW, cosine decay, gradient clipping; no decay on bias/normalization parameters |
| Regularization | Random resized crops, RandAugment, horizontal flips, MixUp/CutMix, label smoothing |
| Class imbalance | Optional inverse-frequency training sampler; report macro F1 and per-class scores |
| Model selection | Best validation macro F1, with lower validation NLL breaking ties; early stopping |
| Calibration | One temperature fitted on validation logits after selection, then applied to test/inference |
| Reproducibility | Seeded splits/loaders, configuration, history, versions, manifest fingerprint |

Defaults follow the selected weights' evaluation preprocessing (384px for EfficientNetV2-S; 224px for ConvNeXt-Tiny). The exact resize, crop, interpolation, and normalization are saved in the checkpoint and reused for evaluation and prediction. Use `--image-size` to trade resolution for memory/speed; this changes the pretrained recipe and requires evaluation.

Useful controls:

```bash
# Reduce GPU memory usage; choose a new output directory for each experiment.
image-classification train --data data/pets --output runs/pets-small --batch-size 4

# Disable augmentations that do not suit your domain.
image-classification train --data data/pets --output runs/pets-plain --no-augment --no-mixup

# A fully offline debugging run.
image-classification --threads 2 train --data data/pets --output runs/pets-debug \
  --model tiny_cnn --epochs 3 --device cpu --no-augment --no-mixup
```

Horizontal flips, rotations, color changes, crops, and mixed samples can change labels in some domains (for example text or subtle color defects). Tune or disable augmentation based on your task. `--no-hflip` disables horizontal flips; `--no-augment` disables the crop/RandAugment pipeline as well. `--no-mixup` disables both MixUp and CutMix. `--workers 0` is the portable default; increase it after checking performance on your machine.

Each run writes:

| File | Contents |
| --- | --- |
| `best.pt` | Best model, class order, preprocessing, fitted temperature, training configuration |
| `splits.json` | Relative image paths, labels, partitions, file/pixel hashes |
| `config.json` | Training options |
| `history.jsonl` | Per-epoch training loss, validation metrics, learning rates, and training stage |
| `metrics.json` | Best epoch, versions, counts, raw/calibrated validation and held-out test metrics |

Run directories must be new or empty to prevent accidental overwrites. Checkpoints are for inference/evaluation; resuming interrupted training is not implemented.

## Evaluate and predict

```bash
image-classification evaluate \
  --checkpoint runs/pets-efficientnet/best.pt \
  --data data/pets \
  --output runs/pets-efficientnet/recheck.json

image-classification predict \
  --checkpoint runs/pets-efficientnet/best.pt \
  --image path/to/photo.jpg \
  --top-k 3

# A directory is processed recursively in batches.
image-classification predict \
  --checkpoint runs/pets-efficientnet/best.pt \
  --image path/to/photos \
  --output predictions.json
```

Evaluation uses the checkpoint's saved manifest (by default beside the checkpoint). `--data` is the original dataset root, and `--manifest` can locate a moved manifest. The default split is `test`; `--split val` is available for diagnostics. Both commands emit JSON, while training progress is printed to stderr. Top-k is capped at the number of classes.

Reports include accuracy, balanced accuracy, macro F1, per-class precision/recall/F1/support, confusion matrix (rows=true, columns=predicted), negative log-likelihood, multiclass Brier score (sum over classes), and 15-bin expected calibration error. Macro F1 includes every configured class; balanced accuracy averages recall over classes present in the evaluated data.

Temperature scaling leaves the predicted class unchanged. It is fitted to reduce **validation** NLL; it does not guarantee better test calibration. `validation_calibrated` is an in-sample diagnostic. Use the untouched `test_calibrated` report to assess generalization, and compare `test` against it. Confidence is relative to known classes; it does not detect unknown objects or distribution shift.

Use validation results to choose models and hyperparameters. Repeatedly choosing models from test results makes the test set part of model selection; preserve a fresh final holdout for serious comparisons. No technique here guarantees a higher score on an unseen dataset.

## Verification

```bash
ruff check .
ruff format --check .
pytest -q
image-classification smoke --output runs/smoke-check
```

Tests cover split reproducibility/leakage, duplicate/corrupt images, class mapping, preprocessing, backbone freezing, metrics, calibration, checkpoint reloads, and the full offline CLI workflow. Both real backbone architectures have offline forward/backward checks; pretrained-weight loading is a separate network-dependent check. GitHub Actions runs the offline checks on Python 3.11 and 3.12.

Additional integration checks:

```bash
# Downloads official ImageNet weights, then fine-tunes/reloads both real backbones on CPU.
pytest -q --run-pretrained

# Requires an environment that permits local multiprocessing IPC sockets.
pytest -q --run-multiworker
```

Multiworker loading uses `spawn` and a 60-second batch timeout. In restricted environments that block local IPC sockets, use the default `--workers 0`. GitHub Actions enables the multiworker check.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the actual checks and limits of this initial implementation.

## References

- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [TorchVision MixUp and CutMix](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html)
- [EfficientNetV2-S weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html)
- [ConvNeXt-Tiny weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
