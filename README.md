# Image Classification Lab

Train and evaluate a single-label image classifier from class folders. The package uses PyTorch and TorchVision, with EfficientNetV2-S and ConvNeXt-Tiny backbones and a small CNN for offline checks.

## Install and try the workflow

Use Python 3.10 or newer in a separate environment:

```bash
git clone https://github.com/AJAHANI16/Image_Classification.git
cd Image_Classification
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
image-classification smoke --output runs/smoke
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Install the matching CUDA build from the [PyTorch installation selector](https://pytorch.org/get-started/locally/) first if you plan to use an NVIDIA GPU.

The smoke command creates synthetic red/blue images, trains the tiny CNN on CPU, reloads its checkpoint, and checks evaluation and prediction. It downloads neither data nor weights. This repository includes no real-task dataset, trained task-specific model, or real-world accuracy result.

## Dataset

For automatic splitting, arrange images by class:

```text
data/pets/cat/image_001.jpg
data/pets/cat/image_002.jpg
data/pets/dog/image_001.jpg
data/pets/dog/image_002.jpg
```

Each class needs at least three unique images to populate train, validation, and test; useful experiments require substantially more. The default split is approximately 70/15/15 and is stratified by class. Alternatively, supply `train/<class>/`, `val/<class>/`, and `test/<class>/` under one root. All three folders must contain the same classes; existing partitions are preserved.

JPEG, PNG, BMP, WebP, and TIFF are supported. Images are EXIF-oriented and converted to RGB. Unreadable files, empty classes, conflicting labels, and exact duplicate decoded images are rejected. A hashed manifest records the split; evaluation rejects changed files or a mismatched manifest.

For images sharing a person, object, video, location, or session, prepare group-separated folders yourself. Exact duplicate checks cannot establish independence between related images.

## Train

```bash
image-classification train \
  --data data/pets \
  --output runs/pets-efficientnet \
  --model efficientnet_v2_s \
  --epochs 15 \
  --batch-size 16
```

Use `--model convnext_tiny` for the other backbone. Pretrained weights download on the first training run. `--no-pretrained` trains from scratch; `--model tiny_cnn` provides an offline debugging path.

The defaults train the classification head for up to two epochs before unfreezing the backbone. AdamW uses a lower backbone learning rate, cosine decay, and gradient clipping. Validation macro F1 selects the checkpoint, with validation negative log-likelihood breaking ties. A temperature fitted on validation logits is then used for held-out test evaluation and inference.

Augmentation includes random crops, RandAugment, horizontal flips, MixUp/CutMix, and label smoothing. Adjust it to your labels: `--no-hflip`, `--no-augment`, and `--no-mixup` disable the corresponding transformations. `--balance-classes` enables inverse-frequency sampling.

Default evaluation preprocessing follows the selected weights: 384px for EfficientNetV2-S and 224px for ConvNeXt-Tiny. `--image-size` overrides resolution. Preprocessing and class order are saved with the checkpoint.

Each new or empty run directory receives `best.pt`, `splits.json`, `config.json`, `history.jsonl`, and `metrics.json`. Checkpoints support inference/evaluation; training resume is not implemented. Load only checkpoints you trust.

## Evaluate and predict

```bash
image-classification evaluate \
  --checkpoint runs/pets-efficientnet/best.pt \
  --data data/pets \
  --output runs/pets-efficientnet/recheck.json

image-classification predict \
  --checkpoint runs/pets-efficientnet/best.pt \
  --image data/pets/cat/image_001.jpg \
  --top-k 3
```

Prediction also accepts a directory and optional `--output predictions.json`. Evaluation uses the saved manifest and defaults to the test split; `--manifest` can locate a moved manifest. Both commands emit JSON.

Reports include accuracy, balanced accuracy, macro F1, per-class scores, confusion matrices, negative log-likelihood, Brier score, and calibration error. Temperature scaling does not change the predicted class and may not improve calibration on unseen data. Confidence is relative to known classes; it does not detect unknown objects or distribution shift. Choose settings with validation data and reserve the final test set for the final comparison.

## Tests and hardware

```bash
ruff check .
ruff format --check .
pytest -q
```

See [testing notes](docs/VALIDATION.md) for optional pretrained-weight and multiprocessing checks. [GitHub Actions](https://github.com/AJAHANI16/Image_Classification/actions) runs the CPU suite. The default device order is CUDA, Apple MPS, then CPU; mixed precision is CUDA-only. `--workers 0` is the portable default. GPU/MPS behavior needs validation on the target hardware.

## References

- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [TorchVision MixUp and CutMix](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html)
- [Temperature scaling](https://arxiv.org/abs/1706.04599)
