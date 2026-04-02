"""Test Pallas fused linear scan kernel on H100."""
import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial


def scan_kernel(h_init_ref, gated_inputs_ref, h_all_ref, *, T: int):
    """Pallas kernel: fused sequential scan.

    Each program processes BLOCK elements of the state vector
    for one (batch*head) pair across all T timesteps.

    The gate (scalar per timestep) is packed into the first element of the
    input vector to avoid a separate small memory transfer.

    Refs:
        h_init_ref: (1, BLOCK) initial state
        gated_inputs_ref: (1, T, BLOCK+1) gate (element 0) + input (elements 1:)
        h_all_ref: (1, T, BLOCK) output states
    """
    state = h_init_ref[0]  # (BLOCK,)

    def body(t, state):
        gi = gated_inputs_ref[0, t]    # (BLOCK+1,)
        gate = gi[0]                    # scalar: the gate
        inp = gi[1:]                    # (BLOCK,): the input
        new_state = gate * state + inp
        h_all_ref[0, t] = new_state
        return new_state

    jax.lax.fori_loop(0, T, body, state)


def pallas_scan(h_init, gates, inputs):
    """Pallas fused linear scan.

    Args:
        h_init: (BH, DD) initial state (pre-flattened)
        gates: (BH, T) scalar gates (pre-transposed)
        inputs: (BH, T, DD) per-timestep inputs (pre-transposed)

    Returns:
        h_all: (BH, T, DD) all states
    """
    BH, T, DD = inputs.shape
    # BLOCK must fit in shared memory: state(BLOCK) + gates(T) + inputs(T*BLOCK) + output(T*BLOCK)
    # Shared mem ≈ 232KB. Need: 2 * T * BLOCK * dtype_size + overhead < 232KB
    # For f32: BLOCK < 232448 / (2 * T * 4) ≈ 232448 / (128 * 4) when T=64
    BLOCK = min(256, pl.next_power_of_2(DD))

    # Pack gate (scalar) into first element of input to avoid small transfer
    # gated_inputs: (BH, T, DD+1) where [:,:,0] = gate, [:,:,1:] = input
    # But DD+1 must be aligned. Pad BLOCK to nearest power of 2 that includes +1.
    BLOCK_PLUS = BLOCK  # We'll use BLOCK that includes the gate

    # Actually, let's just pad the gate into the input array
    gates_expanded = gates[:, :, jnp.newaxis]  # (BH, T, 1)
    gated_inputs = jnp.concatenate([gates_expanded, inputs], axis=-1)  # (BH, T, DD+1)

    # Need BLOCK to cover DD+1 elements
    BLOCK_GI = min(256, pl.next_power_of_2(DD + 1))

    # Pad gated_inputs to BLOCK_GI * cdiv(DD+1, BLOCK_GI) width
    total_width = BLOCK_GI * pl.cdiv(DD + 1, BLOCK_GI)
    if total_width > DD + 1:
        gated_inputs = jnp.pad(gated_inputs,
            ((0,0), (0,0), (0, total_width - DD - 1)))

    return pl.pallas_call(
        partial(scan_kernel, T=T),
        out_shape=jax.ShapeDtypeStruct((BH, T, DD), inputs.dtype),
        grid=(BH, pl.cdiv(DD, BLOCK)),
        in_specs=[
            pl.BlockSpec((1, BLOCK), lambda bh, d: (bh, d * BLOCK)),         # h_init
            pl.BlockSpec((1, T, BLOCK + 1), lambda bh, d: (bh, 0, d * BLOCK)),  # gated_inputs (shifted by d*BLOCK, gate is at offset 0 only for d=0)
        ],
        out_specs=pl.BlockSpec((1, T, BLOCK), lambda bh, d: (bh, 0, d * BLOCK)),
    )(h_init, gated_inputs)


def time_fn(fn, *args, warmup=3, repeats=10):
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    print("Testing Pallas fused scan kernel on H100")
    print(f"Devices: {jax.devices()}")
    print("=" * 80)

    # Test basic Pallas functionality
    print("\n--- Basic Pallas test ---")
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    N = 128
    x = jnp.ones((N,), dtype=jnp.float32)
    y = jnp.ones((N,), dtype=jnp.float32)
    try:
        out = pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct((N,), jnp.float32),
            grid=(1,),
            in_specs=[pl.BlockSpec((N,), lambda i: (0,)),
                      pl.BlockSpec((N,), lambda i: (0,))],
            out_specs=pl.BlockSpec((N,), lambda i: (0,)),
        )(x, y)
        print(f"  Basic Pallas: OK (sum={float(jnp.sum(out))})")
    except Exception as e:
        print(f"  Basic Pallas FAILED: {e}")
        return

    # Test fused scan
    print("\n--- Fused scan test ---")
    BH, T, DD = 4, 4, 256  # small test to debug
    key = jax.random.PRNGKey(0)
    h_init = jax.random.normal(key, (BH, DD), dtype=jnp.float32) * 0.1
    gates = jax.random.uniform(key, (BH, T), dtype=jnp.float32, minval=0.5, maxval=0.99)
    inputs = jax.random.normal(key, (BH, T, DD), dtype=jnp.float32) * 0.1

    try:
        h_all = pallas_scan(h_init, gates, inputs)
        print(f"  Pallas scan shape: {h_all.shape}")

        # Verify correctness
        h = h_init
        for t in range(T):
            h = gates[:, t:t+1] * h + inputs[:, t]
        diff = float(jnp.max(jnp.abs(h_all[:, -1] - h)))
        print(f"  Correctness final state: max_diff={diff:.2e}")

        # Check all timesteps
        h = h_init
        for t in range(T):
            h = gates[:, t:t+1] * h + inputs[:, t]
            diff_t = float(jnp.max(jnp.abs(h_all[:, t] - h)))
            if diff_t > 1e-4 or t < 3:
                print(f"  t={t}: max_diff={diff_t:.2e}  pallas={h_all[0,t,:3]}  expected={h[0,:3]}")
    except Exception as e:
        print(f"  Pallas scan FAILED: {e}")
        return

    # Benchmark
    print("\n--- Benchmark: Pallas vs associative scan ---")
    from atlas_jax.model import linear_scan as assoc_scan

    # Reshape for associative scan API: (B, T, H, D, D)
    B, H, D = 8, 8, 56
    h_init_3d = jax.random.normal(key, (B, H, D, D)) * 0.1
    gates_3d = jax.random.uniform(key, (B, T, H), minval=0.5, maxval=0.99)
    inputs_5d = jax.random.normal(key, (B, T, H, D, D)) * 0.1

    @jax.jit
    def run_assoc(h, g, i):
        return assoc_scan(h, g, i)

    @jax.jit
    def run_pallas(h, g, i):
        return pallas_scan(h, g, i)

    # Flatten for pallas
    h_flat = h_init_3d.reshape(B*H, D*D)
    g_flat = gates_3d.transpose(0, 2, 1).reshape(B*H, T)
    i_flat = inputs_5d.reshape(B, T, H, D*D).transpose(0, 2, 1, 3).reshape(B*H, T, D*D)

    t_assoc, _ = time_fn(run_assoc, h_init_3d, gates_3d, inputs_5d)
    print(f"  Associative scan: {t_assoc*1000:.2f} ms")

    t_pallas, _ = time_fn(run_pallas, h_flat, g_flat, i_flat)
    print(f"  Pallas scan:      {t_pallas*1000:.2f} ms")
    print(f"  Speedup:          {t_assoc/t_pallas:.2f}x")

    # Benchmark at B=32
    print("\n--- Benchmark B=32 ---")
    B2 = 32
    h_init_3d_32 = jax.random.normal(key, (B2, H, D, D)) * 0.1
    gates_3d_32 = jax.random.uniform(key, (B2, T, H), minval=0.5, maxval=0.99)
    inputs_5d_32 = jax.random.normal(key, (B2, T, H, D, D)) * 0.1

    h_flat_32 = h_init_3d_32.reshape(B2*H, D*D)
    g_flat_32 = gates_3d_32.transpose(0, 2, 1).reshape(B2*H, T)
    i_flat_32 = inputs_5d_32.reshape(B2, T, H, D*D).transpose(0, 2, 1, 3).reshape(B2*H, T, D*D)

    t_assoc, _ = time_fn(run_assoc, h_init_3d_32, gates_3d_32, inputs_5d_32)
    print(f"  Associative scan B=32: {t_assoc*1000:.2f} ms")

    t_pallas, _ = time_fn(run_pallas, h_flat_32, g_flat_32, i_flat_32)
    print(f"  Pallas scan B=32:      {t_pallas*1000:.2f} ms")
    print(f"  Speedup:               {t_assoc/t_pallas:.2f}x")

    # bf16 test
    print("\n--- bf16 ---")
    h_flat_bf16 = h_flat_32.astype(jnp.bfloat16)
    g_flat_bf16 = g_flat_32.astype(jnp.bfloat16)
    i_flat_bf16 = i_flat_32.astype(jnp.bfloat16)

    t_pallas_bf16, _ = time_fn(run_pallas, h_flat_bf16, g_flat_bf16, i_flat_bf16)
    print(f"  Pallas scan B=32 bf16: {t_pallas_bf16*1000:.2f} ms")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == '__main__':
    main()
