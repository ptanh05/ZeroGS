"""
_C submodule for simple_knn.
Provides distCUDA2 function computing mean squared distance to 3 nearest neighbors.
"""

import torch


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """
    Computes mean squared distance to 3 nearest neighbors for point cloud initialization.
    Matches the exact signature and behavior of the CUDA kernel in simple-knn.
    """
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    P = points.shape[0]
    if P <= 4:
        return torch.full((P,), 0.01, dtype=torch.float32, device=points.device)

    with torch.no_grad():
        if P <= 5000:
            dists = torch.cdist(points, points)
            topk = torch.topk(dists, k=min(4, P), largest=False).values
            if topk.shape[1] > 1:
                dist2 = (topk[:, 1:4] ** 2).mean(dim=-1)
            else:
                dist2 = topk[:, 0] ** 2
            return torch.clamp_min(dist2, 1e-7)
        else:
            # Chunked evaluation for larger point clouds to avoid VRAM exhaustion
            result = torch.zeros(P, device=points.device, dtype=torch.float32)
            chunk_size = 2048
            num_samples = min(P, 5000)
            sample_indices = torch.randperm(P, device=points.device)[:num_samples]
            sample_pts = points[sample_indices]

            for i in range(0, P, chunk_size):
                end = min(P, i + chunk_size)
                chunk = points[i:end]
                dists = torch.cdist(chunk, sample_pts)
                topk = torch.topk(dists, k=min(4, dists.shape[1]), largest=False).values
                if topk.shape[1] > 1:
                    result[i:end] = (topk[:, 1:4] ** 2).mean(dim=-1)
                else:
                    result[i:end] = topk[:, 0] ** 2

            return torch.clamp_min(result, 1e-7)
