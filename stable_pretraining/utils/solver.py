from typing import Callable, Literal, Optional
from enum import Enum
import torch
from torch import Tensor
import math


class ODESolver(str, Enum):
    """Available ODE solvers ordered roughly by quality/cost tradeoff."""

    EULER = "euler"  # 1st order, 1 NFE/step
    MIDPOINT = "midpoint"  # 2nd order, 2 NFE/step - good for flow matching
    HEUN = "heun"  # 2nd order, 2 NFE/step
    RK4 = "rk4"  # 4th order, 4 NFE/step
    DPM_2 = "dpm_2"  # 2nd order, optimized for diffusion/flow
    DPM_3 = "dpm_3"  # 3rd order, optimized for diffusion/flow


def flow_matching_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    num_steps: int = 50,
    solver: ODESolver | str = ODESolver.DPM_2,
    time_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
    guidance_scale: Optional[float] = None,
    cond_velocity_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    clamp_range: Optional[tuple[float, float]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Sample from a flow matching model via probability flow ODE integration.

    Integrates dx/dt = v(x, t) from t=0 (noise) to t=1 (data).

    :param velocity_fn: Velocity model v(x, t) -> velocity.
        - x: Current state, shape ``shape``
        - t: Time values in [0, 1], shape ``(batch,)``
    :type velocity_fn: Callable[[Tensor, Tensor], Tensor]
    :param shape: Output shape, typically ``(batch, tokens, dim)``.
    :type shape: tuple[int, ...]
    :param num_steps: Number of integration steps (NFE depends on solver).
    :type num_steps: int
    :param solver: ODE solver to use. DPM_2 recommended for best quality/speed.
    :type solver: ODESolver | str
    :param time_schedule: Time discretization schedule.
        - "linear": Uniform spacing (default)
        - "cosine": More steps near t=0 and t=1
        - "quadratic": More steps near t=1 (data)
    :type time_schedule: Literal["linear", "cosine", "quadratic"]
    :param guidance_scale: CFG scale. If set, uses v = v_uncond + scale * (v_cond - v_uncond).
    :type guidance_scale: Optional[float]
    :param cond_velocity_fn: Conditional velocity for CFG. Required if guidance_scale set.
    :type cond_velocity_fn: Optional[Callable[[Tensor, Tensor], Tensor]]
    :param clamp_range: If set, clamp x to this range each step for stability.
    :type clamp_range: Optional[tuple[float, float]]
    :param device: Computation device.
    :type device: Optional[torch.device]
    :param dtype: Computation dtype.
    :type dtype: Optional[torch.dtype]
    :param generator: RNG for reproducibility.
    :type generator: Optional[torch.Generator]
    :param return_trajectory: If True, also return list of intermediate states.
    :type return_trajectory: bool
    :return: Samples of shape ``shape``, optionally with trajectory.
    :rtype: Tensor | tuple[Tensor, list[Tensor]]
    """
    solver = ODESolver(solver) if isinstance(solver, str) else solver
    batch_size = shape[0]

    # Build time schedule
    timesteps = _build_time_schedule(num_steps, time_schedule, device, dtype)

    # Optionally wrap velocity_fn with CFG
    if guidance_scale is not None:
        if cond_velocity_fn is None:
            raise ValueError("cond_velocity_fn required when using guidance_scale")
        velocity_fn = _make_cfg_velocity_fn(
            velocity_fn, cond_velocity_fn, guidance_scale
        )

    # Start from noise
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    trajectory = [x.clone()] if return_trajectory else None

    # Select solver step function
    step_fn = _get_solver_step_fn(solver)

    # Integrate
    for i in range(num_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t

        t_batch = t.expand(batch_size)
        x = step_fn(velocity_fn, x, t_batch, dt, t_next)

        if clamp_range is not None:
            x = x.clamp(*clamp_range)

        if trajectory is not None:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, trajectory
    return x


def _build_time_schedule(
    num_steps: int,
    schedule: str,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    eps=1e-3,
) -> Tensor:
    """Build time discretization from t=0 to t=1."""
    t = torch.linspace(eps, 1 - eps, num_steps + 1, device=device, dtype=dtype)

    if schedule == "linear":
        return t
    elif schedule == "cosine":
        # More steps near boundaries
        return 0.5 * (1 - torch.cos(t * math.pi))
    elif schedule == "quadratic":
        # More steps near t=1 (data distribution)
        return t**2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def _make_cfg_velocity_fn(
    uncond_fn: Callable[[Tensor, Tensor], Tensor],
    cond_fn: Callable[[Tensor, Tensor], Tensor],
    scale: float,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Wrap velocity functions with classifier-free guidance."""

    def guided_velocity(x: Tensor, t: Tensor) -> Tensor:
        v_uncond = uncond_fn(x, t)
        v_cond = cond_fn(x, t)
        return v_uncond + scale * (v_cond - v_uncond)

    return guided_velocity


def _get_solver_step_fn(solver: ODESolver):
    """Return the step function for a given solver."""
    return {
        ODESolver.EULER: _step_euler,
        ODESolver.MIDPOINT: _step_midpoint,
        ODESolver.HEUN: _step_heun,
        ODESolver.RK4: _step_rk4,
        ODESolver.DPM_2: _step_dpm_2,
        ODESolver.DPM_3: _step_dpm_3,
    }[solver]


# === Solver implementations ===


def _step_euler(
    v_fn: Callable, x: Tensor, t: Tensor, dt: Tensor, t_next: Tensor
) -> Tensor:
    """Euler method: x_{n+1} = x_n + dt * v(x_n, t_n)."""
    return x + dt * v_fn(x, t)


def _step_midpoint(
    v_fn: Callable, x: Tensor, t: Tensor, dt: Tensor, t_next: Tensor
) -> Tensor:
    """Midpoint method: Better for flow matching due to straight-path structure."""
    t_mid = t + 0.5 * dt
    x_mid = x + 0.5 * dt * v_fn(x, t)
    return x + dt * v_fn(x_mid, t_mid)


def _step_heun(
    v_fn: Callable, x: Tensor, t: Tensor, dt: Tensor, t_next: Tensor
) -> Tensor:
    """Heun's method (improved Euler / explicit trapezoidal)."""
    v = v_fn(x, t)
    x_euler = x + dt * v
    v_next = v_fn(x_euler, t_next)
    return x + 0.5 * dt * (v + v_next)


def _step_rk4(
    v_fn: Callable, x: Tensor, t: Tensor, dt: Tensor, t_next: Tensor
) -> Tensor:
    """Classic 4th-order Runge-Kutta."""
    half_dt = 0.5 * dt
    t_mid = t + half_dt

    k1 = v_fn(x, t)
    k2 = v_fn(x + half_dt * k1, t_mid)
    k3 = v_fn(x + half_dt * k2, t_mid)
    k4 = v_fn(x + dt * k3, t_next)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _step_dpm_2(
    v_fn: Callable, x: Tensor, t: Tensor, dt: Tensor, t_next: Tensor
) -> Tensor:
    """DPM-Solver-2 adapted for flow matching.

    Based on Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic
    Model Sampling in Around 10 Steps" but adapted for the flow matching ODE
    where paths are (approximately) straight.
    """
    # For flow matching, DPM-2 simplifies to midpoint with a specific coefficient
    r = 0.5  # Can tune this; 0.5 = midpoint
    t_mid = t + r * dt

    v1 = v_fn(x, t)
    x_mid = x + r * dt * v1
    v2 = v_fn(x_mid, t_mid)

    # Linear combination for 2nd order accuracy
    return x + dt * ((1.0 - 0.5 / r) * v1 + (0.5 / r) * v2)


def _step_dpm_3(
    v_fn: Callable, x: Tensor, t: Tensor, dt: Tensor, t_next: Tensor
) -> Tensor:
    """DPM-Solver-3 adapted for flow matching.

    3rd order method using 3 function evaluations.
    """
    r1, r2 = 1.0 / 3.0, 2.0 / 3.0
    t1 = t + r1 * dt
    t2 = t + r2 * dt

    v0 = v_fn(x, t)
    x1 = x + r1 * dt * v0
    v1 = v_fn(x1, t1)
    x2 = x + r2 * dt * v0 + (r2 * (r2 - r1) / (2 * r1)) * dt * (v1 - v0)
    v2 = v_fn(x2, t2)

    # 3rd order combination
    c0 = 1.0 - 1.0 / (2 * r2)
    c1 = 1.0 / (2 * r1 * r2) - 1.0 / (2 * r2 * (r2 - r1))
    c2 = r1 / (2 * r2 * (r2 - r1))  # = 3/4 = 0.75 ✓

    return x + dt * (c0 * v0 + c1 * v1 + c2 * v2)
