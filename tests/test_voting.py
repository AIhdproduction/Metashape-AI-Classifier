import numpy as np

from msam3.config import Classes, GroundAssistConfig, VotingConfig
from msam3.pointcloud.voting import Votes, apply_ground_assist


def test_views_share_and_memmap(tmp_path):
    classes = Classes.model_validate(
        {
            "unknown_code": 1,
            "classes": [
                {"name": "a", "code": 2, "prompts": ["a"]},
                {"name": "b", "code": 64, "prompts": ["b"]},
            ],
        }
    )
    config = VotingConfig(min_views=3, min_share=0.7, max_ram_gb=1e-10)
    votes = Votes(4, classes, config, tmp_path)
    assert isinstance(votes.weights, np.memmap)
    assert votes.memmap_path.exists()
    for _ in range(2):
        votes.add([0, 1, 2, 3], [0, 0, 1, 255], [1, 1, 0.8, 1])
    votes.add([0, 1, 2, 3], [0, 1, 255, 0], [1, 1, 1, 0])
    labels, shares = votes.decide()
    assert labels.tolist() == [2, 1, 1, 1]
    np.testing.assert_allclose(shares, [1.0, 2 / 3, 1.0, 0.0])
    assert votes.views.tolist() == [3, 3, 2, 0]


def test_ground_assist_only_resolves_sam_unknown_points(tmp_path):
    classes = Classes.model_validate(
        {
            "unknown_code": 1,
            "classes": [
                {"name": "ground", "code": 2, "prompts": ["ground"]},
                {"name": "road", "code": 11, "prompts": ["road"]},
                {"name": "sidewalk", "code": 73, "prompts": ["sidewalk"]},
                {"name": "vegetation", "code": 3, "prompts": ["vegetation"]},
            ],
        }
    )
    votes = Votes(4, classes, VotingConfig(min_views=1, min_share=0.7), tmp_path)
    votes.weights[:] = [[0.3, 0.0, 0.0, 0.7], [0.0, 0.2, 0.0, 0.4], [0.0] * 4, [0.0] * 4]
    votes.views[:] = [1, 1, 0, 0]
    labels, shares = votes.decide()

    labels, shares, report = apply_ground_assist(
        labels,
        shares,
        votes,
        np.array([True, True, True, False]),
        GroundAssistConfig(),
    )

    assert labels.tolist() == [3, 11, 2, 1]
    np.testing.assert_allclose(shares, [0.7, 1 / 3, 0.4, 0.0])
    assert report == {"metashape_ground_points": 3, "resolved": {"11": 1, "2": 1}}
