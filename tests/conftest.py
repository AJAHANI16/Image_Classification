from pathlib import Path

import pytest
import torch

from image_classification.smoke import create_demo_data


def pytest_addoption(parser):
    parser.addoption(
        "--run-pretrained", action="store_true", help="Download and check ImageNet weights"
    )
    parser.addoption(
        "--run-multiworker", action="store_true", help="Check multiprocessing data loaders"
    )


def pytest_collection_modifyitems(config, items):
    for marker, flag in (("pretrained", "--run-pretrained"), ("multiworker", "--run-multiworker")):
        if not config.getoption(flag):
            skip = pytest.mark.skip(reason=f"Pass {flag} to enable this integration check")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    create_demo_data(root)
    return root
