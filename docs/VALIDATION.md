# Testing

Run checks from the repository root after installing `.[dev]`:

```bash
ruff check .
ruff format --check .
pytest -q
image-classification smoke --output runs/smoke-check
```

Use a new output directory for each smoke run. The default suite runs offline after dependencies are installed.

## What the tests exercise

- Reproducible, disjoint splits, class mapping, duplicate/corrupt-image handling, and manifest changes.
- Confusion matrices, F1, log-loss, Brier score, and calibration error against known inputs.
- Backbone freezing, BatchNorm state, and forward/backward passes for each architecture.
- Saved preprocessing, checkpoint reload, evaluation, and prediction in a separate process.
- Rejection of invalid arguments and existing non-empty run directories.

## Optional integration checks

```bash
pytest -q --run-pretrained
pytest -q --run-multiworker
```

`--run-pretrained` downloads official ImageNet weights and fine-tunes both backbones on small synthetic images. It checks weight loading and the training path, not performance on photographs.

`--run-multiworker` exercises data loading with multiprocessing. It requires local IPC support; the loader uses `spawn` and a 60-second batch timeout. GitHub Actions enables this check on Python 3.11 and 3.12. See the [run history](https://github.com/AJAHANI16/Image_Classification/actions) for results on a particular revision.

## What remains dataset- and hardware-dependent

The synthetic red/blue task is deliberately easy. Its score establishes that the workflow runs, not that a model generalizes to a real classification task. A representative dataset, independent held-out groups, and task-specific evaluation are needed for that conclusion.

CPU tests do not validate CUDA mixed precision, Apple MPS, GPU memory use, or throughput. Test those paths on the intended hardware. Training resume, distributed training, multilabel classification, detection, and unknown-class rejection are outside the current implementation.
