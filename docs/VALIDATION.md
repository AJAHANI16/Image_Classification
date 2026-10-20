# Initial verification

Verified in this workspace on 2026-10-19. The repository was empty before this implementation.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cpu |
| TorchVision | 0.29.1+cpu |
| NumPy | 2.5.2 |
| Pillow | 12.3.0 |
| pytest | 9.1.1 |
| Ruff | 0.17.1 |

## Results

| Check | Result |
| --- | --- |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `pytest -q --run-pretrained` | 49 passed, 1 skipped, 31.85 seconds |
| Installed `image-classification smoke --output runs/local-smoke` | Passed: training, calibration, checkpoint reload, evaluation, prediction |
| `python -m pip wheel --no-deps .` | Wheel built successfully |
| `python -m pip check` | No broken requirements |
| `python -m compileall -q src tests` | Passed |

The opt-in pretrained tests downloaded the official EfficientNetV2-S and ConvNeXt-Tiny ImageNet weights. Each model completed two training epochs (one frozen-backbone epoch, one unfrozen epoch), augmentation and mixed-label training, validation-only temperature fitting, test evaluation, checkpoint reload, and image prediction. These integration checks used 64px synthetic images to bound CPU time; they establish software operation, not real-task accuracy.

The offline synthetic red/blue demo achieved 100% on its four held-out test images. This intentionally easy, tiny task validates the workflow only. It is not evidence of accuracy on photographs, deployment data, or unknown classes.

## Coverage and limits

- Splits are reproducible, class-stratified, disjoint, and preserved for pre-split folders. Exact duplicate decoded RGB content and conflicting labels are rejected. Near-duplicates and correlated subjects require user-provided group-separated splits.
- Metrics are checked against hand-computed confusion matrices, F1, negative log-likelihood, Brier score, and calibration error. Temperature fitting is checked for validation-NLL improvement and unchanged predicted classes.
- Evaluation preprocessing is compared numerically with both official weights' transforms. Checkpoint reloads preserve class order and preprocessing. Edited datasets and mismatched manifests are rejected.
- Backbone freezing is checked for both gradients and BatchNorm running statistics. All three model architectures complete offline forward/backward checks.
- Full CLI smoke/prediction checks run in separate Python processes. Invalid arguments fail with a concise error; existing runs cannot be overwritten.
- The local environment blocks the IPC sockets PyTorch uses for multiworker data loaders. That integration test is opt-in and was skipped locally. The portable default is `--workers 0`; worker mode uses spawn and a bounded timeout. GitHub Actions is configured to run `--run-multiworker` on Python 3.11 and 3.12.
- CUDA mixed precision, CUDA hardware, and Apple MPS were not available for local validation. No GPU/MPS performance or compatibility result is claimed.
- No real dataset, trained task-specific weights, or real-world benchmark was supplied. Select augmentations and validate accuracy/calibration on your own representative data before deployment. Training resume, distributed training, multilabel classification, detection, and unknown-class rejection are outside this initial implementation.

Reproduce the local checks from the repository root using the commands in the README. The default dependency ranges permit compatible newer packages; each training run records the actual Torch/TorchVision versions used.
