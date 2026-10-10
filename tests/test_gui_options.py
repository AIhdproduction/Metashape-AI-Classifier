from msam3.gui_options import (
    batch_time_factor,
    estimate_seconds,
    historical_seconds_per_image,
    maximum_safe_batch,
    settings_override,
    tile_count,
)
from msam3.gui_worker import settings_for_session


def test_start_window_override_is_applied_only_to_the_session(tmp_path):
    (tmp_path / "options.json").write_text(
        """
        {
          "settings_override": {
            "sam": {"image_scale": 1.0, "tile_overlap": 0.3, "prompt_batch_size": 12},
            "voting": {"min_views": 2},
            "ground_assist": {"enabled": false},
            "postprocess": {
              "enabled": true,
              "vegetation": {"enabled": false},
              "road": {"enabled": false}
            },
            "smoothing": {"enabled": true}
          }
        }
        """,
        encoding="utf-8",
    )

    settings = settings_for_session(tmp_path)

    assert settings.sam.image_scale == 1.0
    assert settings.sam.tile_overlap == 0.3
    assert settings.sam.prompt_batch_size == 12
    assert settings.voting.min_views == 2
    assert not settings.ground_assist.enabled
    assert not settings.postprocess.vegetation.enabled
    assert not settings.postprocess.road.enabled
    assert settings.smoothing.enabled


def test_estimate_reflects_image_quality_and_overlap():
    quick = estimate_seconds(291, 0.5, 0.1, 1, baseline=50)
    accurate = estimate_seconds(291, 1.0, 0.3, 2, baseline=50)
    larger_batch = estimate_seconds(291, 0.75, 0.2, 1, baseline=50, prompt_batch_size=16)
    standard_batch = estimate_seconds(291, 0.75, 0.2, 1, baseline=50, prompt_batch_size=8)

    assert quick > 0
    assert accurate > quick
    assert larger_batch < standard_batch
    assert settings_override(0.75, 0.2, 8, 1, True, True, True, True, False)["sam"] == {
        "image_scale": 0.75,
        "tile_overlap": 0.2,
        "prompt_batch_size": 8,
    }


def test_auto_batch_reserves_gpu_headroom_and_never_exceeds_sixteen():
    assert maximum_safe_batch(24.0, 24.0, 0.75, 0.20) == 16
    assert maximum_safe_batch(24.0, 24.0, 1.0, 0.30) == 4
    assert maximum_safe_batch(16.0, 16.0, 0.75, 0.20) == 4
    assert maximum_safe_batch(10.0, 16.0, 0.75, 0.20) is None


def test_estimate_uses_exact_tile_count_and_normalizes_previous_profile(tmp_path):
    session = tmp_path / "old"
    session.mkdir()
    (session / "session.json").write_text(
        '{"project": "D:/Test Script/Test_script.psx"}', encoding="utf-8"
    )
    (session / "job.json").write_text(
        '{"settings": {"sam": {"image_scale": 1.0, "tile_overlap": 0.2, "prompt_batch_size": 4}}}',
        encoding="utf-8",
    )
    lines = [
        f"2026-10-09 08:{minute:02d}:00,000 INFO Chunk 1: image {index}/9 (image-{index}) complete"
        for index, minute in enumerate(range(1, 10), start=1)
    ]
    (session / "stage2.log").write_text("\n".join(lines), encoding="utf-8")
    sizes = [(3000, 2000)]

    baseline = historical_seconds_per_image(tmp_path, sizes, "D:/Test Script/Test_script.psx")
    expected = 60 * tile_count(3000, 2000, 0.75, 0.2) / tile_count(3000, 2000, 1.0, 0.2)
    expected /= batch_time_factor(4)

    assert baseline == expected
    quick = estimate_seconds(1, 0.5, 0.1, 1, baseline, 8, sizes)
    accurate = estimate_seconds(1, 1.0, 0.3, 1, baseline, 8, sizes)
    assert accurate > quick
