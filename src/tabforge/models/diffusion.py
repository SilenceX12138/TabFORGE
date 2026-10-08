"""Noise schedules and sampling for the Score-based Diffusion Transformer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


def power_mean_schedule(
    num_steps: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a descending Karras schedule with a terminal zero.

    Args:
        num_steps: Number of non-zero noise levels.
        sigma_min: Lowest non-zero noise level.
        sigma_max: Starting noise level.
        rho: Power-law curvature parameter.
        device: Device on which to create the schedule.

    The non-zero portion contains ``num_steps`` values from ``sigma_max`` to
    ``sigma_min``.  The extra zero is the final clean-data transition used by
    the Euler/Heun sampler.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    nonzero = (max_inv + ramp * (min_inv - max_inv)).pow(rho)
    return torch.cat([nonzero, nonzero.new_zeros(1)])


def edm_coefficients(sigma: torch.Tensor, sigma_data: float = 1.0) -> dict[str, torch.Tensor]:
    """Compute EDM input, output, skip, and noise coefficients.

    Args:
        sigma: Per-row noise levels.
        sigma_data: Expected clean-data standard deviation.
    """
    sigma = torch.as_tensor(sigma)
    sigma_data_tensor = torch.as_tensor(sigma_data, dtype=sigma.dtype, device=sigma.device)
    sigma2 = sigma.square()
    data2 = sigma_data_tensor.square()
    return {
        "c_skip": data2 / (sigma2 + data2),
        "c_out": sigma * sigma_data_tensor / (sigma2 + data2).sqrt(),
        "c_in": 1 / (sigma2 + data2).sqrt(),
        "c_noise": sigma.log() / 4,
    }


def edm_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 1.0,
    *,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the EDM denoising objective averaged over rows.

    Args:
        prediction: Denoised latent prediction.
        target: Clean latent target with the same shape.
        sigma: Per-row noise levels.
        sigma_data: Expected clean-data standard deviation.
        loss_mask: Optional token mask where ``True`` contributes to loss.

    EDM weights each row by ``(sigma² + sigma_data²) /
    (sigma * sigma_data)²`` and sums the token/dimension errors within that
    row.  Keeping the row reduction explicit preserves the scale used by the
    migrated checkpoint.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}")
    if prediction.ndim < 1 or prediction.shape[0] != sigma.reshape(-1).shape[0]:
        raise ValueError("sigma must contain one value per prediction row")
    sigma = sigma.reshape(-1).to(device=prediction.device, dtype=prediction.dtype)
    sigma_data_tensor = prediction.new_tensor(float(sigma_data))
    weight = (sigma.square() + sigma_data_tensor.square()) / (sigma * sigma_data_tensor).square().clamp_min(
        torch.finfo(prediction.dtype).eps
    )
    row_error = _diffusion_row_error(prediction, target, loss_mask)
    return (weight * row_error).mean()


def _diffusion_row_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Sum full errors or scale unknown-token errors to the full grid.

    Args:
        prediction: Denoised latent prediction.
        target: Clean latent target.
        loss_mask: Optional token-selection mask.
    """
    squared = (prediction - target).square()
    if loss_mask is None:
        return squared.reshape(len(prediction), -1).sum(dim=1)
    token_error = squared.sum(dim=-1) * loss_mask.to(squared.dtype)
    unknown_count = loss_mask.sum(dim=1).to(squared.dtype)
    return token_error.sum(dim=1) * prediction.shape[1] / unknown_count


class EDMPreconditioner(nn.Module):
    """EDM preconditioning around the Score-based Diffusion Transformer."""

    def __init__(self, denoiser: nn.Module, sigma_data: float = 1.0) -> None:
        """Wrap a denoiser with EDM preconditioning.

        Args:
            denoiser: Noise-conditioned latent denoising module.
            sigma_data: Expected clean-data standard deviation.
        """
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = float(sigma_data)

    def forward(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply EDM scaling around one denoiser prediction.

        Args:
            noisy: Noisy latent grid.
            sigma: Per-row noise levels.
            observation_mask: Optional internal observed-token mask.
        """
        coefficients = edm_coefficients(sigma, self.sigma_data)
        scaled = noisy * coefficients["c_in"].reshape(-1, 1, 1)
        prediction = self.denoiser(scaled, sigma, observation_mask)
        return coefficients["c_skip"].reshape(-1, 1, 1) * noisy + coefficients["c_out"].reshape(-1, 1, 1) * prediction


@dataclass(frozen=True)
class _SamplingOptions:
    """Group sampler dependencies shared by every noise transition."""

    denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
    observation: torch.Tensor | None
    observation_mask: torch.Tensor | None
    generator: torch.Generator | None
    enable_heun_correction: bool
    sigma_churn: float


@torch.no_grad()
def euler_heun_sample(
    denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    initial: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    observation: torch.Tensor | None = None,
    observation_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    enable_heun_correction: bool = False,
    sigma_churn: float = 0.0,
) -> torch.Tensor:
    """Run the Score-based Diffusion Transformer sampler.

    Args:
        denoiser: EDM-preconditioned denoising callable.
        initial: Initial noisy latent grid.
        sigmas: Descending schedule with a terminal zero.
        observation: Optional clean latent grid to clamp.
        observation_mask: Token mask where ``True`` means observed.
        generator: Optional device random stream.
        enable_heun_correction: Apply a second-order correction per transition.
        sigma_churn: Stochastic noise injected across the schedule.

    ``observation_mask=True`` means that a token is observed.  Such tokens are
    clamped before the first denoising call, after every Euler/Heun transition,
    and after the terminal transition.  The public imputer converts its
    ``True means regenerate`` mask to this internal polarity at the boundary.
    """
    _validate_sampling_inputs(initial, sigmas, observation, observation_mask)
    options = _SamplingOptions(
        denoiser,
        observation,
        observation_mask,
        generator,
        enable_heun_correction,
        sigma_churn,
    )
    result = _clamp_observation(initial, observation, observation_mask)
    for index in range(len(sigmas) - 1):
        result = _sampling_transition(result, sigmas, index, options)
    return result


def _validate_sampling_inputs(
    initial: torch.Tensor,
    sigmas: torch.Tensor,
    observation: torch.Tensor | None,
    observation_mask: torch.Tensor | None,
) -> None:
    """Validate schedule and observation shapes at the sampler boundary.

    Args:
        initial: Initial noisy latent grid.
        sigmas: Candidate noise schedule.
        observation: Optional clean latent grid.
        observation_mask: Optional observed-token mask.
    """
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("sigmas must contain at least a start and end value")
    if observation is not None and observation.shape != initial.shape:
        raise ValueError("observation must have the same shape as initial")
    if observation_mask is not None and observation_mask.shape != initial.shape[:2]:
        raise ValueError("observation_mask must have shape (batch, n_tokens)")
    if (observation is None) != (observation_mask is None):
        raise ValueError("observation and observation_mask must be provided together")


def _sampling_transition(
    result: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
    options: _SamplingOptions,
) -> torch.Tensor:
    """Apply one churn, Euler, and optional Heun transition.

    Args:
        result: Current latent grid.
        sigmas: Complete sampling schedule.
        index: Current schedule position.
        options: Shared sampler dependencies and settings.
    """
    sigma = sigmas[index].expand(len(result))
    sigma_next = sigmas[index + 1]
    result, sigma_hat = _apply_churn(result, sigma, len(sigmas), options)
    result = _clamp_observation(result, options.observation, options.observation_mask)
    denoised = options.denoiser(result, sigma_hat, options.observation_mask)
    derivative = (result - denoised) / sigma_hat.clamp_min(1e-8).reshape(-1, 1, 1)
    result_euler = result + (sigma_next - sigma_hat).reshape(-1, 1, 1) * derivative
    if options.enable_heun_correction and float(sigma_next) > 0:
        result = _heun_transition(result, result_euler, derivative, sigma_hat, sigma_next, options)
    else:
        result = result_euler
    return _clamp_observation(result, options.observation, options.observation_mask)


def _apply_churn(
    result: torch.Tensor,
    sigma: torch.Tensor,
    schedule_length: int,
    options: _SamplingOptions,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject configured stochastic churn before a denoising transition.

    Args:
        result: Current latent grid.
        sigma: Current per-row noise levels.
        schedule_length: Number of schedule entries.
        options: Shared sampler dependencies and settings.
    """
    if options.sigma_churn <= 0:
        return result, sigma
    gamma = min(options.sigma_churn / max(schedule_length - 1, 1), 2**0.5 - 1)
    sigma_hat = sigma * (1 + gamma)
    noise = torch.randn(
        result.shape,
        device=result.device,
        dtype=result.dtype,
        generator=options.generator,
    )
    scale = (sigma_hat.square() - sigma.square()).clamp_min(0).sqrt().reshape(-1, 1, 1)
    return result + scale * noise, sigma_hat


def _heun_transition(
    result: torch.Tensor,
    result_euler: torch.Tensor,
    derivative: torch.Tensor,
    sigma_hat: torch.Tensor,
    sigma_next: torch.Tensor,
    options: _SamplingOptions,
) -> torch.Tensor:
    """Correct an Euler proposal using a second denoising evaluation.

    Args:
        result: Latents before the Euler proposal.
        result_euler: First-order proposal.
        derivative: Derivative at the current noise level.
        sigma_hat: Noise level after churn.
        sigma_next: Next schedule noise level.
        options: Shared sampler dependencies and settings.
    """
    proposal = _clamp_observation(result_euler, options.observation, options.observation_mask)
    next_denoised = options.denoiser(proposal, sigma_next.expand(len(result)), options.observation_mask)
    next_derivative = (proposal - next_denoised) / sigma_next.clamp_min(1e-8)
    step = (sigma_next - sigma_hat).reshape(-1, 1, 1)
    return result + step * (derivative + next_derivative) / 2


def _clamp_observation(
    result: torch.Tensor,
    observation: torch.Tensor | None,
    observation_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Restore observed tokens after a stochastic sampler operation.

    Args:
        result: Candidate latent grid.
        observation: Optional clean latent grid.
        observation_mask: Token mask where ``True`` means observed.
    """
    if observation is None:
        return result
    return torch.where(observation_mask.unsqueeze(-1), observation, result)
