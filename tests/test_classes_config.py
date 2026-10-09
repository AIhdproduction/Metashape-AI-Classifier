import copy

import pytest

from msam3.config import Classes, Settings, load_classes, load_settings


def test_default_codes_and_append():
    classes = load_classes()
    custom = {c.name: c.code for c in classes.classes if c.code >= 64}
    assert custom == {
        "sidewalk": 64,
        "curb": 65,
        "vehicle": 66,
        "construction_machinery": 67,
        "utility_pole": 68,
        "traffic_sign": 69,
        "fence": 70,
        "scaffolding": 71,
        "construction_material": 72,
    }
    data = classes.model_dump()
    data["classes"].append({"name": "rock", "prompts": ["rock"]})
    assert Classes.model_validate(data).classes[-1].code == 73


def test_postprocess_has_a_rule_for_every_default_class():
    settings = load_settings()
    classes = load_classes()
    assert set(settings.postprocess.class_rules) == {item.name for item in classes.classes}


@pytest.mark.parametrize(
    "change",
    [
        {"code": 256},
        {"code": -1},
        {"code": 1},
        {"code": 2.5},
        {"prompts": []},
        {"prompts": [" "]},
    ],
)
def test_invalid_class(change):
    data = {"classes": [{"name": "ground", "code": 2, "prompts": ["ground"], **change}]}
    with pytest.raises(ValueError):
        Classes.model_validate(data)


def test_duplicates_and_reserved_explicit_codes():
    data = {
        "classes": [{"name": "a", "prompts": ["a"]}, {"name": "b", "code": 64, "prompts": ["b"]}]
    }
    assert [c.code for c in Classes.model_validate(data).classes] == [65, 64]
    duplicate = copy.deepcopy(data)
    duplicate["classes"][0]["name"] = "b"
    with pytest.raises(ValueError, match="names"):
        Classes.model_validate(duplicate)
    data["classes"][0]["code"] = 64
    with pytest.raises(ValueError, match="codes"):
        Classes.model_validate(data)


def test_exhausted_codes_and_config_typos():
    with pytest.raises(ValueError, match="No free"):
        Classes.model_validate(
            {
                "custom_code_start": 255,
                "classes": [
                    {"name": "a", "code": 255, "prompts": ["a"]},
                    {"name": "b", "prompts": ["b"]},
                ],
            }
        )
    with pytest.raises(ValueError):
        Settings.model_validate({"voting": {"min_vews": 2}})
