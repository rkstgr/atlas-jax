#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=bench_bf16
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

set -euo pipefail
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
echo "==> GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python -c "
import jax, jax.numpy as jnp, equinox as eqx, optax, time
from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

# 100M config: 12L, 768d, 12 heads, dim_head=64, expand=2
results = []
for bf16, fused, batch in [
    (False, False, 22),   # f32 nonfused
    (False, True,  16),   # f32 fused
    (True,  False, 44),   # bf16 nonfused (2x batch fits)
    (True,  True,  32),   # bf16 fused
]:
    jax.config.update('jax_default_matmul_precision', 'high' if bf16 else 'float32')
    cfg = AtlasConfig(
        vocab_size=256, n_layer=12, n_head=12, n_embd=768,
        memory_expand=2, chunk_size=64, ns_steps=5, omega_window=2,
        poly_degree=2, deep_memory=True, pe_ste=True,
        fused_chunk=fused, dropout=0.0, gate_bias_init=0.0, max_lr=0.1,
        logit_softcap=0.0, stop_grad_chunks=True, geglu_ff=True,
        num_persist_mem_tokens=4, use_checkpoint=True)

    key = jax.random.PRNGKey(42)
    model = Atlas(cfg, key=key, pad_vocab_size_to=1)
    if bf16:
        model = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype==jnp.float32 else x,
            model, is_leaf=eqx.is_array)
    n = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    opt = optax.adamw(1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    idx = jax.random.randint(key, (batch, 1024), 0, 256)
    tgt = jnp.roll(idx, -1, axis=1)

    def _upcast(g):
        return jax.tree.map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) and x.dtype==jnp.bfloat16 else x, g, is_leaf=eqx.is_array)

    @eqx.filter_jit
    def step(m, o, x, t):
        def lf(m):
            l, _ = m(x)
            f = l.reshape(-1, l.shape[-1]).astype(jnp.float32)
            tf = t.reshape(-1)
            return -jnp.mean(jax.nn.log_softmax(f, axis=-1)[jnp.arange(tf.shape[0]), tf])
        loss, g = eqx.filter_value_and_grad(lf)(m)
        g = _upcast(g)
        u, no = opt.update(eqx.filter(g, eqx.is_array), o, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, u), no, loss

    # Warmup
    for _ in range(3):
        model, opt_state, loss = step(model, opt_state, idx, tgt); float(loss)

    # Benchmark
    ts = []
    for _ in range(10):
        t0 = time.time(); model, opt_state, loss = step(model, opt_state, idx, tgt); float(loss)
        ts.append(time.time() - t0)

    avg = sum(ts) / len(ts)
    pk = jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9
    tps = batch * 1024 / avg
    prec = 'bf16' if bf16 else 'f32 '
    mode = 'FUSED' if fused else 'NOFUS'
    print(f'RESULT | 100M {prec} {mode} | B={batch:>2} | {avg*1000:>7.0f}ms | {pk:>5.1f}GB | {tps:>5.0f} tok/s')
    jax.clear_caches()
    import gc; gc.collect()

print('Done')
"
