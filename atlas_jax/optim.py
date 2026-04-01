"""Muon + AdamW optimizer as optax GradientTransformations.

Muon: MomentUm Orthogonalized by Newton-schulz
- Nesterov momentum
- Polar Express orthogonalization on gradient
- NorMuon variance reduction (per-neuron adaptive scaling)
- Cautious weight decay

Combined via optax.multi_transform:
- 2D weight matrices -> Muon
- Embeddings -> AdamW
- Scalars/gates/conv weights -> AdamW

LR schedule: linear warmup + constant + linear warmdown.

Reference: https://kellerjordan.github.io/posts/muon/
NorMuon: arXiv 2510.05491
"""

from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import optax

from atlas_jax.polar_express import polar_express, POLAR_EXPRESS_COEFFS


# ---------------------------------------------------------------------------
# Muon state and transform
# ---------------------------------------------------------------------------

class MuonState(NamedTuple):
    momentum: jax.Array        # first moment buffer
    second_moment: jax.Array   # factored second moment for NorMuon


def muon(
    learning_rate: float = 0.02,
    momentum: float = 0.95,
    beta2: float = 0.9,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
) -> optax.GradientTransformation:
    """Muon optimizer: momentum -> polar_express -> NorMuon variance reduction -> cautious update.

    Designed for 2D weight matrices only.
    """

    def init_fn(params):
        # Determine reduction dimension for NorMuon: reduce along the larger dim
        shape = params.shape
        assert params.ndim == 2, f"Muon requires 2D params, got shape {shape}"
        if shape[-2] >= shape[-1]:
            # Tall/square: reduce along columns -> per-row variance
            sm_shape = (shape[-2], 1)
        else:
            # Wide: reduce along rows -> per-column variance
            sm_shape = (1, shape[-1])
        return MuonState(
            momentum=jnp.zeros_like(params),
            second_moment=jnp.zeros(sm_shape, dtype=params.dtype),
        )

    def update_fn(updates, state, params=None):
        grad = updates
        dtype = grad.dtype
        shape = grad.shape

        # LR scaling for tall matrices (same convention as PyTorch)
        lr_scale = max(1.0, shape[-2] / shape[-1]) ** 0.5

        # Nesterov momentum
        new_momentum = state.momentum * momentum + grad * (1 - momentum)
        g = grad * (1 - momentum) + new_momentum * momentum

        # Polar Express orthogonalization (in f32 for stability)
        X = g.astype(jnp.float32)
        frob_norm = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True))
        X = X / (frob_norm * 1.01 + 1e-6)

        if shape[-2] > shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X.T @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X @ X.T
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X.astype(dtype)

        # NorMuon variance reduction
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        red_dim_size = shape[red_dim]

        v_mean = jnp.mean(g.astype(jnp.float32) ** 2, axis=red_dim, keepdims=True)
        v_norm_sq = jnp.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = jnp.sqrt(v_norm_sq)

        new_second_moment = state.second_moment * beta2 + v_mean.astype(dtype) * (1 - beta2)
        step_size = jax.lax.rsqrt(jnp.maximum(new_second_moment, 1e-10).astype(jnp.float32))

        scaled_sq_sum = (v_mean * red_dim_size) * step_size ** 2
        v_norm_new = jnp.sqrt(jnp.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))
        g = g * final_scale.astype(dtype)

        # Cautious weight decay + update
        if params is not None and weight_decay > 0:
            mask = (g * params) >= 0
            update = -(learning_rate * lr_scale * g + learning_rate * lr_scale * weight_decay * params * mask)
        else:
            update = -learning_rate * lr_scale * g

        return update, MuonState(momentum=new_momentum, second_moment=new_second_moment)

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def warmup_constant_warmdown_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    warmdown_steps: int = 0,
    min_lr: float = 0.0,
) -> optax.Schedule:
    """Linear warmup -> constant -> linear warmdown."""
    def schedule(step):
        # Warmup phase
        warmup_ratio = jnp.where(warmup_steps > 0, step / warmup_steps, 1.0)
        warmup_lr = min_lr + (peak_lr - min_lr) * jnp.minimum(warmup_ratio, 1.0)

        # Warmdown phase
        steps_left = total_steps - step
        warmdown_ratio = jnp.where(
            warmdown_steps > 0,
            jnp.minimum(steps_left / warmdown_steps, 1.0),
            1.0,
        )
        warmdown_lr = min_lr + (peak_lr - min_lr) * warmdown_ratio

        # Combine: use warmup during warmup phase, warmdown during warmdown, constant otherwise
        lr = jnp.where(step < warmup_steps, warmup_lr, jnp.where(steps_left < warmdown_steps, warmdown_lr, peak_lr))
        return lr

    return schedule


# ---------------------------------------------------------------------------
# Combined optimizer builder
# ---------------------------------------------------------------------------

def build_optimizer(
    model,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.004,
    scalar_lr: float = 0.05,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    total_steps: int = 1,
    warmdown_steps: int = 0,
):
    """Build combined Muon + AdamW optimizer for Atlas.

    Parameter grouping:
    - 2D weight matrices in blocks -> Muon
    - Embedding weights -> AdamW
    - Everything else (conv, gates, poly_coeffs) -> AdamW with different lr

    Returns:
        (optimizer, opt_state) tuple ready for use with eqx.filter_jit.
    """
    # For now return a simple AdamW as the multi_transform setup requires
    # careful pytree label mapping. We'll refine this.
    # TODO: implement proper param grouping with optax.multi_transform
    optimizer = optax.adamw(learning_rate=matrix_lr, weight_decay=weight_decay)
    return optimizer
