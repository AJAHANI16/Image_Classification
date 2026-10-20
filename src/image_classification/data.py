"""Audited image folders and reproducible splits, with no stochastic evaluation transforms."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    # Resolving also rejects symbolic links that escape the dataset root.
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Invalid relative image path: {relative}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes the dataset root: {relative}")
    return path


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    sha256: str
    pixel_sha256: str
    split: str


@dataclass
class Manifest:
    classes: list[str]
    samples: list[Sample]
    seed: int
    schema_version: int = 1

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def counts(self) -> dict[str, dict[str, int]]:
        counts = Counter((sample.split, sample.label) for sample in self.samples)
        return {
            split: {name: counts[split, label] for label, name in enumerate(self.classes)}
            for split in SPLITS
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            classes=raw["classes"],
            samples=[Sample(**sample) for sample in raw["samples"]],
            seed=raw["seed"],
            schema_version=raw["schema_version"],
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported split manifest version")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("At least two distinct classes are required")
        paths: set[str] = set()
        digests: set[str] = set()
        for sample in self.samples:
            if sample.split not in SPLITS or not 0 <= sample.label < len(self.classes):
                raise ValueError("Invalid label or split in manifest")
            if sample.path in paths or sample.pixel_sha256 in digests:
                raise ValueError("Duplicate path or image content in manifest")
            if Path(sample.path).is_absolute() or ".." in Path(sample.path).parts:
                raise ValueError(f"Invalid relative image path: {sample.path}")
            paths.add(sample.path)
            digests.add(sample.pixel_sha256)
        for split, counts in self.counts().items():
            if not all(counts.values()):
                raise ValueError(f"Every class needs at least one image in {split}")

    def verify_files(self, root: Path, split: str | None = None) -> None:
        for sample in self.samples:
            if split is None or sample.split == split:
                path = safe_path(root, sample.path)
                if not path.is_file() or file_sha256(path) != sample.sha256:
                    raise ValueError(f"Dataset changed since splitting: {sample.path}")


def _class_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))


def build_manifest(
    root: Path, *, seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> Manifest:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")

    explicit = any((root / split).is_dir() for split in SPLITS)
    if explicit and not all((root / split).is_dir() for split in SPLITS):
        raise ValueError("Provide all three train/val/test folders, or only class folders")
    classes = _class_names(root / "train" if explicit else root)
    if len(classes) < 2:
        raise ValueError("At least two class folders are required")
    if explicit and any(_class_names(root / split) != classes for split in SPLITS):
        raise ValueError("train/val/test must contain exactly the same class folders")

    seen_pixels: dict[str, tuple[str, int]] = {}
    samples: list[Sample] = []
    rng = random.Random(seed)
    for split in SPLITS if explicit else ("unsplit",):
        directory = root / split if explicit else root
        for label, name in enumerate(classes):
            paths = image_paths(directory / name)
            if not paths:
                raise ValueError(f"No supported images in {directory / name}")
            if not explicit:
                if len(paths) < 3:
                    raise ValueError(
                        f"Class {name!r} needs at least 3 unique images for three splits"
                    )
                rng.shuffle(paths)
                n_val = max(1, round(len(paths) * val_fraction))
                n_test = max(1, round(len(paths) * test_fraction))
                if n_val + n_test >= len(paths):
                    raise ValueError(f"Too few training images for class {name!r}; lower fractions")
            for index, path in enumerate(paths):
                relative = path.relative_to(root).as_posix()
                safe_path(root, relative)
                try:
                    image = read_rgb(path)
                    pixel_hash = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()
                except (OSError, ValueError) as error:
                    raise ValueError(f"Cannot decode image {relative}: {error}") from error
                if pixel_hash in seen_pixels:
                    previous, old_label = seen_pixels[pixel_hash]
                    reason = "conflicting labels" if old_label != label else "duplicate content"
                    raise ValueError(
                        f"{reason}: {previous} and {relative}. Remove duplicates first."
                    )
                seen_pixels[pixel_hash] = (relative, label)
                assigned = split
                if not explicit:
                    assigned = (
                        "val" if index < n_val else ("test" if index < n_val + n_test else "train")
                    )
                samples.append(Sample(relative, label, file_sha256(path), pixel_hash, assigned))
    manifest = Manifest(classes, sorted(samples, key=lambda sample: sample.path), seed)
    manifest.validate()
    return manifest


class ImageDataset(Dataset):
    def __init__(self, root: Path, manifest: Manifest, split: str, transform: Callable):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.root = root.resolve()
        self.samples = [sample for sample in manifest.samples if sample.split == split]
        self.transform = transform

    @property
    def labels(self) -> list[int]:
        return [sample.label for sample in self.samples]

    def sampling_weights(self) -> torch.Tensor:
        counts = Counter(self.labels)
        return torch.tensor([1.0 / counts[label] for label in self.labels], dtype=torch.double)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        return self.transform(read_rgb(safe_path(self.root, sample.path))), sample.label
