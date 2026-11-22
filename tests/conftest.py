import pytest
import torch

from msam3 import user_classes
from msam3.app import i18n


@pytest.fixture(autouse=True)
def cuda_is_never_queried(monkeypatch):
    """Tests see a PC without CUDA unless they inject a fake one. The real GPU is never asked."""

    def query(*args, **kwargs):
        raise AssertionError("a test queried the real GPU")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_properties", query)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", query)


@pytest.fixture(autouse=True)
def isolated_user_files(tmp_path_factory, monkeypatch):
    """No test may read or write this PC's own classes and settings."""
    directory = tmp_path_factory.mktemp("user")
    monkeypatch.setattr(user_classes, "USER_CLASSES", directory / "classes.user.yml")
    monkeypatch.setattr(i18n, "USER_SETTINGS", directory / "user_settings.json")
    return directory
