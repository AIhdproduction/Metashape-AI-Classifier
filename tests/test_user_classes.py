import pytest

from msam3 import backends, user_classes
from msam3.config import REPO_ROOT, load_classes
from msam3.user_classes import UserClassError, UserClasses


@pytest.fixture
def standard():
    return load_classes()


def test_no_edits_are_exactly_the_standard_classes(standard):
    assert user_classes.apply(standard, UserClasses()) == standard


def test_edits_are_saved_next_to_the_untouched_standard_file(standard):
    untouched = (REPO_ROOT / "config/classes.yml").read_bytes()
    edits = user_classes.save_class(standard, UserClasses(), "Schacht Öl", ["oil pit"], "#12ab34")
    edits = user_classes.set_prompts(standard, edits, "ground", ["gravel", "soil"])
    edits = user_classes.set_enabled(standard, edits, "water", False)
    user_classes.write(edits)

    loaded = backends.MODELS["sam3"].classes()

    names = [item.name for item in loaded.classes]
    assert names[-1] == "schacht_oel" and "water" not in names
    assert loaded.classes[-1].code == 120
    assert loaded.classes[0].prompts == ["gravel", "soil"]
    assert names[:-1] == [item.name for item in standard.classes if item.name != "water"]
    assert (REPO_ROOT / "config/classes.yml").read_bytes() == untouched
    assert user_classes.apply(standard, UserClasses()) == standard


def test_auto_code_stays_below_128_and_skips_used_codes(standard):
    edits = UserClasses()
    for index in range(12):
        edits = user_classes.save_class(standard, edits, f"own_{index}", ["thing"], "#123456")
    codes = [item.code for item in edits.custom]
    assert codes[:8] == list(range(120, 128))
    assert len(set(codes)) == len(codes) and max(codes) < 128
    assert not set(codes) & {item.code for item in standard.classes}


@pytest.mark.parametrize(
    "name,prompts,color,code,key",
    [
        ("ground", ["x"], "#112233", None, "exists"),
        ("1x", ["x"], "#112233", None, "name"),
        ("x", [], "#112233", None, "prompts"),
        ("x", ["x"], "red", None, "color"),
        ("x", ["x"], "#112233", 128, "code_range"),
        ("x", ["x"], "#112233", 1, "code_range"),
        ("x", ["x"], "#112233", 11, "code_used"),
    ],
)
def test_editor_rejects_invalid_classes(standard, name, prompts, color, code, key):
    with pytest.raises(UserClassError) as raised:
        user_classes.save_class(standard, UserClasses(), name, prompts, color, code)
    assert raised.value.key == key


def test_a_disabled_classs_code_stays_reserved(standard):
    edits = user_classes.set_enabled(standard, UserClasses(), "water", False)
    with pytest.raises(UserClassError, match="already used"):
        user_classes.save_class(standard, edits, "pond", ["pond"], "#112233", 9)
    restored = user_classes.set_enabled(standard, edits, "water", True)
    assert user_classes.apply(standard, restored) == standard


def test_edit_remove_and_last_class(standard):
    edits = user_classes.save_class(standard, UserClasses(), "a", ["a"], "#112233")
    edits = user_classes.save_class(standard, edits, "b", ["b"], "#445566", 100, original="a")
    assert [(item.name, item.code) for item in edits.custom] == [("b", 100)]
    assert not user_classes.remove_class(standard, edits, "b").custom
    everything = UserClasses(disabled=[item.name for item in standard.classes])
    with pytest.raises(UserClassError, match="at least one|At least one"):
        user_classes.apply(standard, everything)


def test_broken_user_file_is_reported_and_does_not_block_flurseg():
    user_classes.USER_CLASSES.write_text("custom: [", encoding="utf-8")
    with pytest.raises(UserClassError, match="Cannot read"):
        backends.MODELS["sam3"].classes()
    assert len(backends.MODELS["flurseg"].classes().classes) == 11


def test_prompt_parsing():
    assert user_classes.parse_prompts(" manhole, manhole cover;\npit, manhole ") == [
        "manhole",
        "manhole cover",
        "pit",
    ]


@pytest.mark.parametrize("color", ["#112233\n", "#1122334", "#112233 "])
def test_colour_must_match_completely(standard, color):
    with pytest.raises(UserClassError) as raised:
        user_classes.save_class(standard, UserClasses(), "x", ["x"], color)
    assert raised.value.key == "color"


def test_name_must_match_completely(standard, monkeypatch):
    monkeypatch.setattr(user_classes, "slug", lambda text: text)  # slug() would strip the newline.
    with pytest.raises(UserClassError) as raised:
        user_classes.save_class(standard, UserClasses(), "manhole\n", ["x"], "#112233")
    assert raised.value.key == "name"


@pytest.mark.parametrize("name", ["no_such_class", "manhole"])
def test_prompts_and_switch_only_for_standard_classes(standard, name):
    edits = user_classes.save_class(standard, UserClasses(), "manhole", ["manhole"], "#112233")
    for change in (
        lambda: user_classes.set_prompts(standard, edits, name, ["cover"]),
        lambda: user_classes.set_enabled(standard, edits, name, False),
    ):
        with pytest.raises(UserClassError) as raised:
            change()
        assert raised.value.key == "not_standard"
