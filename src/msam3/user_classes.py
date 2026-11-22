"""The user's SAM 3 class edits, stored as a diff next to config/classes.yml (never rewritten)."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from msam3.cache import replace_file
from msam3.config import REPO_ROOT, Classes, SemanticClass, StrictModel

USER_CLASSES = REPO_ROOT / "config/classes.user.yml"
NAME = re.compile(r"[a-z][a-z0-9_]*")
COLOR = re.compile(r"#[0-9A-Fa-f]{6}")
MAX_IMPORT_CODE = 127
# Codes 0 and 1 mean "created, never classified" and "unclassified" in Metashape.
MIN_CLASS_CODE = 2
TRANSLITERATION = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", chr(0xDF): "ss"})


class UserClassError(ValueError):
    """An edit the editor rejects; ``key`` selects the translated message."""

    MESSAGES = {
        "name": "Invalid class name {name}: use a-z, digits and underscores, start with a letter",
        "exists": "The class name {name} already exists",
        "prompts": "The class {name} needs at least one English prompt",
        "color": "Invalid colour {color}: use #RRGGBB",
        "code_range": "LAS code {code} is outside {low}..{high}",
        "code_used": "LAS code {code} is already used",
        "no_code": "No free LAS code below 128 remains",
        "none_left": "At least one class must stay enabled",
        "not_standard": "{name} is not a standard class",
        "file": "Cannot read the user classes {path}: {detail}",
    }

    def __init__(self, key: str, **values):
        self.key, self.values = key, values
        super().__init__(self.MESSAGES[key].format(**values))


class UserClasses(StrictModel):
    version: Literal[1] = 1
    disabled: list[str] = Field(default_factory=list)
    prompts: dict[str, list[str]] = Field(default_factory=dict)
    custom: list[SemanticClass] = Field(default_factory=list)


def read(path: Path | None = None) -> UserClasses:
    path = path or USER_CLASSES
    if not path.is_file():
        return UserClasses()
    try:
        return UserClasses.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as error:
        raise UserClassError("file", path=path, detail=str(error)) from error


def write(edits: UserClasses, path: Path | None = None) -> None:
    path = path or USER_CLASSES
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = yaml.safe_dump(edits.model_dump(), sort_keys=False, allow_unicode=True)
    temporary.write_text("# SAM 3 class edits, written by the app.\n" + text, encoding="utf-8")
    replace_file(temporary, path)


def slug(text: str) -> str:
    name = text.strip().lower().translate(TRANSLITERATION)
    return re.sub(r"[\s-]+", "_", name)


def parse_prompts(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[,;\n]", text)]
    return list(dict.fromkeys(part for part in parts if part))


def free_code(standard: Classes, edits: UserClasses, ignore: str | None = None) -> int:
    used = _used_codes(standard, edits, ignore)
    start = standard.custom_code_start
    for code in [*range(start, MAX_IMPORT_CODE + 1), *range(64, min(start, MAX_IMPORT_CODE + 1))]:
        if code not in used:
            return code
    raise UserClassError("no_code")


def _used_codes(standard: Classes, edits: UserClasses, ignore: str | None = None) -> set[int]:
    # Disabled standard classes keep their code so that enabling them again cannot collide.
    used = {standard.unknown_code, *(item.code for item in standard.classes)}
    return used | {
        item.code for item in edits.custom if item.name != ignore and item.code is not None
    }


def apply(standard: Classes, edits: UserClasses) -> Classes:
    """Standard classes keep their names, codes and order; own classes are appended."""
    reserved = {standard.unknown_code, *(item.code for item in standard.classes)}
    standard_names = {item.name for item in standard.classes}
    result = []
    for item in standard.classes:
        if item.name in edits.disabled:
            continue
        data = item.model_dump()
        if item.name in edits.prompts:
            data["prompts"] = [text for text in edits.prompts[item.name] if text.strip()]
            if not data["prompts"]:
                raise UserClassError("prompts", name=item.name)
        result.append(data)
    for item in edits.custom:
        if item.name in standard_names:
            raise UserClassError("exists", name=item.name)
        if not item.prompts:
            raise UserClassError("prompts", name=item.name)
        code = item.code if item.code is not None else free_code(standard, edits, item.name)
        if not MIN_CLASS_CODE <= code <= MAX_IMPORT_CODE:
            raise UserClassError("code_range", code=code, low=MIN_CLASS_CODE, high=MAX_IMPORT_CODE)
        if code in reserved:
            raise UserClassError("code_used", code=code)
        reserved.add(code)
        result.append({**item.model_dump(), "code": code})
    if not result:
        raise UserClassError("none_left")
    return Classes.model_validate({**standard.model_dump(), "classes": result})


def save_class(
    standard: Classes,
    edits: UserClasses,
    name: str,
    prompts: list[str],
    color: str,
    code: int | None = None,
    original: str | None = None,
) -> UserClasses:
    """Return edits with one own class added, or ``original`` replaced."""
    name = slug(name)
    if not NAME.fullmatch(name):
        raise UserClassError("name", name=name)
    others = [item for item in edits.custom if item.name != original]
    if name in {item.name for item in (*standard.classes, *others)}:
        raise UserClassError("exists", name=name)
    if not prompts:
        raise UserClassError("prompts", name=name)
    if not COLOR.fullmatch(color):
        raise UserClassError("color", color=color)
    if code is None:
        code = free_code(standard, edits, original)
    elif not MIN_CLASS_CODE <= code <= MAX_IMPORT_CODE:
        raise UserClassError("code_range", code=code, low=MIN_CLASS_CODE, high=MAX_IMPORT_CODE)
    elif code in _used_codes(standard, edits, original):
        raise UserClassError("code_used", code=code)
    new = SemanticClass(name=name, code=code, prompts=prompts, color=color)
    custom = [new if item.name == original else item for item in edits.custom]
    if new not in custom:
        custom.append(new)
    return edits.model_copy(update={"custom": custom})


def remove_class(standard: Classes, edits: UserClasses, name: str) -> UserClasses:
    changed = edits.model_copy(
        update={"custom": [item for item in edits.custom if item.name != name]}
    )
    apply(standard, changed)
    return changed


def set_enabled(standard: Classes, edits: UserClasses, name: str, enabled: bool) -> UserClasses:
    _standard_class(standard, name)
    disabled = [item for item in edits.disabled if item != name]
    if not enabled:
        disabled.append(name)
    changed = edits.model_copy(update={"disabled": disabled})
    apply(standard, changed)
    return changed


def set_prompts(
    standard: Classes, edits: UserClasses, name: str, prompts: list[str]
) -> UserClasses:
    original = _standard_class(standard, name).prompts
    if not prompts:
        raise UserClassError("prompts", name=name)
    changed = dict(edits.prompts)
    changed.pop(name, None)
    if prompts != original:
        changed[name] = prompts
    return edits.model_copy(update={"prompts": changed})


def _standard_class(standard: Classes, name: str) -> SemanticClass:
    for item in standard.classes:
        if item.name == name:
            return item
    raise UserClassError("not_standard", name=name)
