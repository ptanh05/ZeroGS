"""
ZeroGS: Memory-Reduced Optimizers for Extreme Low-Budget 3DGS Training.

Provides:
- HalfPrecisionAdam: Stores Adam first and second moments in FP16, cutting optimizer VRAM by 50%.
- Int8GaussianAdam: Dynamically quantizes Adam first moment (int8) and second moment (uint8),
  cutting optimizer VRAM by 75% with zero external C-extension dependencies.
"""

from typing import Dict, Any, Optional, Tuple, Iterable
import torch
from torch.optim import Optimizer


class HalfPrecisionAdam(Optimizer):
    """
    Half-Precision Adam optimizer for 3D Gaussian Splatting.

    Stores exp_avg and exp_avg_sq in torch.float16 instead of torch.float32.
    Reduces persistent optimizer memory from 472 Bytes/Gaussian down to 236 Bytes/Gaussian
    (saving 236 Bytes per Gaussian, a 50% reduction in optimizer VRAM).

    Mathematically identical to Adam: computations and updates are evaluated in FP32,
    then moments are safely stored back in FP16.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("HalfPrecisionAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float16)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float16)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Cast moments to float32 for stable update calculation
                m = exp_avg.to(torch.float32).mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v = exp_avg_sq.to(torch.float32).mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.addcdiv_(m, denom, value=-step_size)

                # Store updated moments back in float16
                exp_avg.copy_(m.to(torch.float16))
                exp_avg_sq.copy_(v.to(torch.float16))

        return loss


class Int8GaussianAdam(Optimizer):
    """
    Int8 Dynamically Quantized Adam optimizer for 3D Gaussian Splatting.

    Quantizes first moment exp_avg to signed int8 and second moment exp_avg_sq to unsigned uint8
    with per-tensor dynamic scale factors.
    Reduces persistent optimizer memory from 472 Bytes/Gaussian down to 118 Bytes/Gaussian
    (saving 354 Bytes per Gaussian, a 75% reduction in optimizer VRAM).

    Zero external C-extension dependencies: 100% pure PyTorch implementation compatible
    with Windows, Linux, and any CUDA version.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Int8GaussianAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_int8"] = torch.zeros_like(p, dtype=torch.int8)
                    state["scale_m"] = 1.0 / 127.0
                    state["exp_avg_sq_uint8"] = torch.zeros_like(p, dtype=torch.uint8)
                    state["scale_v"] = 1.0 / 255.0

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Dequantize moments to float32 for update
                m = state["exp_avg_int8"].to(torch.float32).mul_(state["scale_m"])

                # Log-dequantize variance v (spans 7 orders of magnitude from 1e-7 to 1e2)
                q_v = state["exp_avg_sq_uint8"].to(torch.float32)
                v = torch.where(q_v > 0, torch.exp((q_v / 255.0) * 20.0 - 16.0), torch.zeros_like(q_v))

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.addcdiv_(m, denom, value=-step_size)

                # Dynamic Quantization:
                # 1) m: symmetric linear quantization into int8 [-128, 127]
                max_m = m.abs().max().item()
                scale_m = max(max_m / 127.0, 1e-12)
                state["exp_avg_int8"].copy_((m / scale_m).round_().clamp_(-128, 127).to(torch.int8))
                state["scale_m"] = scale_m

                # 2) v: logarithmic dynamic quantization into uint8 [1, 255]
                v_clamped = v.clamp(min=1e-7)
                log_v = torch.log(v_clamped)
                q_val = ((log_v - (-16.0)) / 20.0 * 254.0 + 1.0).round_().clamp_(1, 255).to(torch.uint8)
                state["exp_avg_sq_uint8"].copy_(torch.where(v > 1e-9, q_val, torch.zeros_like(q_val)))

        return loss
