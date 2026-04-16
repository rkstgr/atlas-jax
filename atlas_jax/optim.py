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

from atlas_jax.memory_layer import polar_express, POLAR_EXPRESS_COEFFS


# ---------------------------------------------------------------------------
# Muon state and transform
# ---------------------------------------------------------------------------

class MuonState(NamedTuple):
    momentum: jax.Array        # first moment buffer
    second_moment: jax.Array   # factored second moment for NorMuon


def _mT(x):
    """Matrix transpose: swap last two dims (works for 2D and batched 3D+)."""
    return jnp.swapaxes(x, -2, -1)


def _muon_leaf_init(param, momentum_val, beta2_val):
    """Init state for a single parameter (2D or batched 3D for stacked layers)."""
    shape = param.shape
    if shape[-2] >= shape[-1]:
        sm_shape = (*shape[:-2], shape[-2], 1)
    else:
        sm_shape = (*shape[:-2], 1, shape[-1])
    return MuonState(
        momentum=jnp.zeros_like(param),
        second_moment=jnp.zeros(sm_shape, dtype=param.dtype),
    )


def _muon_leaf_update(grad, state, param, learning_rate, momentum, beta2,
                       weight_decay, ns_steps):
    """Update a single 2D parameter with Muon."""
    dtype = grad.dtype
    shape = grad.shape

    lr_scale = max(1.0, shape[-2] / shape[-1]) ** 0.5

    # Nesterov momentum
    new_momentum = state.momentum * momentum + grad * (1 - momentum)
    g = grad * (1 - momentum) + new_momentum * momentum

    # Polar Express orthogonalization (in f32)
    X = g.astype(jnp.float32)
    frob_norm = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True))
    X = X / (frob_norm * 1.01 + 1e-6)

    if shape[-2] > shape[-1]:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = _mT(X) @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ _mT(X)
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
    if param is not None and weight_decay > 0:
        mask = (g * param) >= 0
        update = -(learning_rate * lr_scale * g + learning_rate * lr_scale * weight_decay * param * mask)
    else:
        update = -learning_rate * lr_scale * g

    return update, MuonState(momentum=new_momentum, second_moment=new_second_moment)


def muon(
    learning_rate: float = 0.02,
    momentum: float = 0.95,
    beta2: float = 0.9,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
) -> optax.GradientTransformation:
    """Muon optimizer: momentum -> polar_express -> NorMuon -> cautious update.

    Works on pytrees of 2D weight matrices (compatible with multi_transform).
    """

    def init_fn(params):
        return jax.tree.map(
            lambda p: _muon_leaf_init(p, momentum, beta2), params)

    def update_fn(updates, state, params=None):
        if params is None:
            params = jax.tree.map(lambda _: None, updates)

        flat_grads, treedef = jax.tree.flatten(updates)
        flat_states = jax.tree.flatten(state, is_leaf=lambda x: isinstance(x, MuonState))[0]
        flat_params = jax.tree.flatten(params)[0]

        new_updates, new_states = [], []
        for g, s, p in zip(flat_grads, flat_states, flat_params):
            u, ns = _muon_leaf_update(g, s, p, learning_rate, momentum, beta2,
                                       weight_decay, ns_steps)
            new_updates.append(u)
            new_states.append(ns)

        return (jax.tree.unflatten(treedef, new_updates),
                jax.tree.unflatten(treedef, new_states))

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

def _label_params(model):
    """Label each parameter for multi_transform routing.

    Labels:
    - 'muon': 2D weight matrices in blocks (projections, gate weights)
    - 'embedding': wte.weight
    - 'lm_head': lm_head.weight
    - 'scalar': everything else (conv weights/biases, poly_coeffs, 1D params)

    Returns a pytree with the same structure as eqx.filter(model, eqx.is_array),
    but with string labels replacing arrays.
    """
    import equinox as eqx

    params = eqx.filter(model, eqx.is_array)

    # Use flatten_with_path to get (keypath, leaf) pairs including None leaves
    flat_with_path, treedef = jax.tree_util.tree_flatten_with_path(
        params, is_leaf=lambda x: x is None)

    labels = []
    for key_path, leaf in flat_with_path:
        if leaf is None:
            labels.append(None)
            continue

        path_str = jax.tree_util.keystr(key_path)

        if 'wte' in path_str and 'weight' in path_str:
            labels.append('embedding')
        elif 'lm_head' in path_str and 'weight' in path_str:
            labels.append('lm_head')
        elif ('blocks' in path_str and 'weight' in path_str
              and 'conv' not in path_str
              and hasattr(leaf, 'ndim') and leaf.ndim >= 2):
            # Linear weight matrices in blocks → Muon.
            # With scan-over-layers, these are ndim 3 (n_layer, out, in).
            # Conv weights excluded ('conv' in path) — they go to scalar.
            labels.append('muon')
        else:
            labels.append('scalar')

    return jax.tree.unflatten(treedef, labels)


def build_optimizer(
    model,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.004,
    lm_head_lr: float = 0.004,
    scalar_lr: float = 0.05,
    muon_wd: float = 0.0,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    warmup_steps: int = 0,
    total_steps: int = 1,
    warmdown_steps: int = 0,
    max_grad_norm: float = 1.0,
    n_embd: int = 512,
):
    """Build combined Muon + AdamW optimizer for Atlas.

    Parameter grouping (matching nanochat reference):
    - 2D weight matrices in blocks -> Muon (orthogonalized momentum)
    - Embedding weights -> AdamW (beta1=0.8, beta2=0.995, wd=0.001)
    - LM head weights -> AdamW (beta1=0.8, beta2=0.96, wd=0.01)
    - Everything else (conv, gates, poly_coeffs) -> AdamW (beta1=0.8, beta2=0.95, wd=0.0)

    Returns:
        (optimizer, param_labels) — optimizer and the label pytree for multi_transform.
    """
    import equinox as eqx

    # dmodel scaling for embedding/lm_head LRs (nanochat convention)
    dmodel_scale = (n_embd / 768) ** -0.5

    # LR schedule: linear warmup -> constant -> linear warmdown
    lr_schedule = warmup_constant_warmdown_schedule(
        peak_lr=1.0,  # multiplied by per-group LR
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        warmdown_steps=warmdown_steps,
        min_lr=0.05,  # 5% of peak at end
    )

    # Per-group transforms
    muon_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        muon(
            learning_rate=matrix_lr,
            momentum=muon_momentum,
            weight_decay=muon_wd,
            ns_steps=muon_ns_steps,
        ),
        optax.scale_by_schedule(lr_schedule),
    )

    embedding_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=embedding_lr * dmodel_scale,
            b1=0.8, b2=0.995, eps=1e-10,
            weight_decay=0.001,
        ),
        optax.scale_by_schedule(lr_schedule),
    )

    lm_head_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=lm_head_lr * dmodel_scale,
            b1=0.8, b2=0.96, eps=1e-10,
            weight_decay=0.01,
        ),
        optax.scale_by_schedule(lr_schedule),
    )

    scalar_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=scalar_lr,
            b1=0.8, b2=0.95, eps=1e-10,
            weight_decay=0.0,
        ),
        optax.scale_by_schedule(lr_schedule),
    )

    # Pre-compute flat labels from model structure.
    # Can't pass the label pytree directly to multi_transform because
    # Equinox modules are callable, and optax would treat it as a function.
    _label_tree = _label_params(model)
    _flat_labels, _label_treedef = jax.tree.flatten(
        _label_tree, is_leaf=lambda x: x is None or isinstance(x, str))

    def _label_fn(params):
        """Reconstruct label pytree matching params structure."""
        _, params_treedef = jax.tree.flatten(
            params, is_leaf=lambda x: x is None)
        return jax.tree.unflatten(params_treedef, _flat_labels)

    optimizer = optax.multi_transform(
        transforms={
            'muon': muon_tx,
            'embedding': embedding_tx,
            'lm_head': lm_head_tx,
            'scalar': scalar_tx,
        },
        param_labels=_label_fn,
    )

    return optimizer, _label_fn
