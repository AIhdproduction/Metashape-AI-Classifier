"""All visible texts: German (Swiss spelling, the default) and English."""

import json
import logging
from functools import cache
from pathlib import Path

from msam3.cache import atomic_json
from msam3.config import REPO_ROOT

LOG = logging.getLogger(__name__)
LANGUAGES = {"de": "Deutsch", "en": "English"}
DEFAULT_LANGUAGE = "de"
LOCALES = Path(__file__).resolve().parent / "locales"
USER_SETTINGS = REPO_ROOT / "config/user_settings.json"


@cache
def catalogue(language: str) -> dict[str, str]:
    texts = {}

    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}.{key}" if prefix else key, item)
        else:
            texts[prefix] = value

    flatten("", json.loads((LOCALES / f"{language}.json").read_text(encoding="utf-8")))
    return texts


def load_language(path: Path | None = None) -> str:
    path = path or USER_SETTINGS
    try:
        language = json.loads(path.read_text(encoding="utf-8"))["language"]
    except FileNotFoundError:
        return DEFAULT_LANGUAGE
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOG.warning("Cannot read %s (%r); using German", path, error)
        return DEFAULT_LANGUAGE
    return language if language in LANGUAGES else DEFAULT_LANGUAGE


def save_language(language: str, path: Path | None = None) -> None:
    atomic_json(path or USER_SETTINGS, {"language": language})


class Translator:
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language

    def __call__(self, key: str, **values) -> str:
        text = catalogue(self.language).get(key) or catalogue("en").get(key, key)
        return text.format(**values) if values else text

    def class_name(self, name: str) -> str:
        key = f"class.{name.lower()}"
        # Own classes have only the name the user gave them.
        return catalogue(self.language).get(key, name)

    def number(self, value: int) -> str:
        separator = "’" if self.language == "de" else ","
        return f"{int(value):,}".replace(",", separator)
