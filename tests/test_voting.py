import numpy as np

from msam3.config import Classes, VotingConfig
from msam3.pointcloud.voting import Votes


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
