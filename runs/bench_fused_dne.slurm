#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=bench_fused
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

set -euo pipefail
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python -c "
import jax, jax.numpy as jnp, equinox as eqx, optax, time
jax.config.update('jax_default_matmul_precision', 'float32')
from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

for expand, fused in [(1, False), (1, True), (2, False), (2, True)]:
    cfg = AtlasConfig(
        vocab_size=256, n_layer=8, n_head=4, n_embd=320, dim_head=64,
        chunk_size=64, ns_steps=5, omega_window=2, poly_degree=2,
        deep_memory=True, memory_expand=expand, pe_ste=True,
        fused_chunk=fused, dropout=0.0, gate_bias_init=0.0,
        max_lr=0.1, logit_softcap=0.0, stop_grad_chunks=True,
        geglu_ff=True, num_persist_mem_tokens=4)

    key = jax.random.PRNGKey(42)
    model = Atlas(cfg, key=key, pad_vocab_size_to=1)
    n = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(2e-4))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    idx = jax.random.randint(key, (4, 512), 0, 256)
    targets = jnp.roll(idx, -1, axis=1)

    @eqx.filter_jit
    def step(model, opt_state, x, t):
        def loss_fn(m):
            logits, _ = m(x)
            lf = logits.reshape(-1, logits.shape[-1])
            tf = t.reshape(-1)
            return -jnp.mean(jax.nn.log_softmax(lf, axis=-1)[jnp.arange(tf.shape[0]), tf])
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt = opt.update(eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt, loss

    # Warmup
    model, opt_state, loss = step(model, opt_state, idx, targets)
    float(loss)

    # Benchmark (fwd+bwd+optim)
    times = []
    for _ in range(20):
        t0 = time.time()
        model, opt_state, loss = step(model, opt_state, idx, targets)
        float(loss)
        times.append(time.time() - t0)

    avg = sum(times[5:]) / len(times[5:])  # skip first 5 for warmth
    mem = jax.devices()[0].memory_stats()
    peak_gb = mem['peak_bytes_in_use'] / 1e9

    tag = 'FUSED' if fused else 'NOFUSE'
    print(f'expand={expand} {tag}: {avg*1000:.0f}ms/step, peak={peak_gb:.2f}GB, params={n/1e6:.1f}M')
    jax.clear_caches()
    del model, opt_state
"
echo "==> Done"
