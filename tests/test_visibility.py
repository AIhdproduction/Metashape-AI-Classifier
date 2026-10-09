import torch

from msam3.config import VisibilityConfig
from msam3.pointcloud.visibility import ZBuffer


def test_two_parallel_planes_hidden_across_batches():
    buffer = ZBuffer(
        20,
        20,
        VisibilityConfig(zbuffer_scale=1, splat_px=0, rel_tolerance=0, abs_tolerance_m=0.01),
        "cpu",
    )
    yy, xx = torch.meshgrid(torch.arange(5, 15), torch.arange(5, 15), indexing="ij")
    uv = torch.stack((xx.flatten(), yy.flatten()), dim=1).float()
    near, far = torch.full((len(uv),), 5.0), torch.full((len(uv),), 10.0)
    buffer.add(uv, far)
    buffer.add(uv, near)
    assert buffer.visible(uv, near).all()
    assert not buffer.visible(uv, far).any()


def test_splat_and_tolerance():
    config = VisibilityConfig(
        zbuffer_scale=0.5, splat_px=1, rel_tolerance=0.02, abs_tolerance_m=0.05
    )
    buffer = ZBuffer(10, 10, config, "cpu")
    buffer.add(torch.tensor([[4.0, 4.0]]), torch.tensor([10.0]))
    assert buffer.visible(torch.tensor([[6.0, 4.0]]), torch.tensor([10.2])).item()
    assert not buffer.visible(torch.tensor([[6.0, 4.0]]), torch.tensor([10.3])).item()
