import pytest
import torch


def test_cuda_rasterizer_physical_stats():
    """Verifies that the compiled ZeroGS CUDA rasterizer extracts in-situ statistics."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")

    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = "cuda"
    P = 100
    H, W = 128, 128

    means3D = torch.randn(P, 3, device=device) * 0.4
    means3D[:, 2] += 2.5  # place in front of camera
    means2D = torch.zeros(P, 3, device=device, requires_grad=True)
    opacities = torch.rand(P, 1, device=device) * 0.7 + 0.3
    scales = torch.ones(P, 3, device=device) * 0.08
    rotations = torch.zeros(P, 4, device=device)
    rotations[:, 0] = 1.0
    shs = torch.zeros(P, 16, 3, device=device)
    shs[:, 0, :] = 1.0

    bg = torch.zeros(3, device=device)
    W_mat = torch.eye(4, device=device)
    P_mat = torch.eye(4, device=device)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=W_mat,
        projmatrix=P_mat,
        sh_degree=3,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=shs,
        scales=scales,
        rotations=rotations,
    )

    assert len(out) == 6, f"Expected 6 returned values from ZeroGS rasterizer, got {len(out)}"
    color, radii, depth, mean_T, depth_var, vis_count = out

    # Tensor shape checks
    assert color.shape == (3, H, W)
    assert radii.shape == (P,)
    assert mean_T.shape == (P,)
    assert depth_var.shape == (P,)
    assert vis_count.shape == (P,)

    # Physical boundary checks
    assert torch.all(mean_T >= 0.0) and torch.all(mean_T <= 1.0), "mean_T must be bounded in [0, 1]"
    assert torch.all(vis_count >= 0), "vis_count must be non-negative"
    assert vis_count.sum() > 0, "At least some pixels should be touched by visible Gaussians"

    # Backward gradient flow check
    loss = color.sum()
    loss.backward()
    assert means2D.grad is not None
    assert means2D.grad.norm().item() > 0.0
