from msam3.gui_options import estimate_seconds, maximum_safe_batch, settings_override
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
