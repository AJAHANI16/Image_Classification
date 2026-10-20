import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from image_classification.data import Manifest, build_manifest, read_rgb, safe_path


def test_splits_are_disjoint_stratified_and_reproducible(dataset_root, tmp_path):
    manifest = build_manifest(dataset_root, seed=7)
    assert manifest.fingerprint == build_manifest(dataset_root, seed=7).fingerprint
    assert manifest.fingerprint != build_manifest(dataset_root, seed=8).fingerprint
    assert manifest.classes == ["blue", "red"]
    sets = {
        split: {sample.path for sample in manifest.samples if sample.split == split}
        for split in ("train", "val", "test")
    }
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["val"]
    assert not sets["val"] & sets["test"]
    assert len(set.union(*sets.values())) == 24
    assert manifest.counts() == {
        "train": {"blue": 8, "red": 8},
        "val": {"blue": 2, "red": 2},
        "test": {"blue": 2, "red": 2},
    }
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert Manifest.load(path).fingerprint == manifest.fingerprint
    manifest.verify_files(dataset_root)


def test_presplit_data_preserves_assignments(dataset_root, tmp_path):
    root = tmp_path / "presplit"
    for split, indices in (("train", range(8)), ("val", range(8, 10)), ("test", range(10, 12))):
        for name in ("blue", "red"):
            (root / split / name).mkdir(parents=True)
            for index in indices:
                source = dataset_root / name / f"{index:03}.png"
                shutil.copy(source, root / split / name / source.name)
    manifest = build_manifest(root)
    assert all(sample.path.split("/")[0] == sample.split for sample in manifest.samples)
    assert manifest.counts()["train"] == {"blue": 8, "red": 8}
    (root / "test" / "red").rename(root / "test" / "other")
    with pytest.raises(ValueError, match="same class"):
        build_manifest(root)


@pytest.mark.parametrize("different_class", [False, True])
def test_decoded_duplicates_and_conflicting_labels_are_rejected(dataset_root, different_class):
    target = dataset_root / ("red" if different_class else "blue") / "copy.bmp"
    # Different encoding, identical RGB pixels: byte-only duplicate detection would miss this.
    read_rgb(dataset_root / "blue" / "000.png").save(target)
    with pytest.raises(ValueError, match="conflicting labels|duplicate content"):
        build_manifest(dataset_root)


def test_corrupt_supported_image_is_not_silently_skipped(dataset_root):
    (dataset_root / "red" / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot decode image"):
        build_manifest(dataset_root)


def test_changed_image_is_detected_even_if_still_readable(dataset_root):
    manifest = build_manifest(dataset_root)
    sample = next(sample for sample in manifest.samples if sample.split == "test")
    Image.new("RGB", (40, 48), "white").save(dataset_root / sample.path)
    with pytest.raises(ValueError, match="Dataset changed"):
        manifest.verify_files(dataset_root, "test")


@pytest.mark.parametrize("relative", ["../outside.png", "/tmp/outside.png"])
def test_path_traversal_is_rejected(dataset_root, relative):
    with pytest.raises(ValueError, match="Invalid relative"):
        safe_path(dataset_root, relative)


def test_symlink_cannot_escape_dataset(dataset_root, tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (32, 32), "black").save(outside)
    link = dataset_root / "blue" / "outside.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links unavailable on this platform")
    with pytest.raises(ValueError, match="escapes"):
        build_manifest(dataset_root)


@pytest.mark.parametrize("fractions", [(0, 0.15), (0.8, 0.3), (float("nan"), 0.2)])
def test_bad_split_fractions_are_rejected(dataset_root, fractions):
    with pytest.raises(ValueError, match="fractions"):
        build_manifest(dataset_root, val_fraction=fractions[0], test_fraction=fractions[1])


def test_too_few_images_are_rejected(dataset_root):
    for path in sorted((dataset_root / "blue").glob("*.png"))[2:]:
        path.unlink()
    with pytest.raises(ValueError, match="at least 3"):
        build_manifest(dataset_root)


def test_incomplete_explicit_splits_are_rejected(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match="all three"):
        build_manifest(tmp_path)


def test_invalid_manifest_schema_and_labels_are_rejected(dataset_root, tmp_path):
    path = tmp_path / "manifest.json"
    build_manifest(dataset_root).save(path)
    payload = json.loads(path.read_text())
    payload["samples"][0]["label"] = 100
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid label"):
        Manifest.load(path)
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        Manifest.load(path)


def test_grayscale_and_rgba_inputs_become_rgb(tmp_path: Path):
    for mode in ("L", "RGBA"):
        path = tmp_path / f"{mode}.png"
        Image.new(mode, (17, 21)).save(path)
        image = read_rgb(path)
        assert image.mode == "RGB"
        assert image.size == (17, 21)
