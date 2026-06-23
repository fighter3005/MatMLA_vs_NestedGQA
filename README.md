# MatMLA vs Nested GQA

A from-scratch small-LM training pipeline that supports:

- **Baseline MHA** and **baseline GQA** transformers (RMSNorm + RoPE + SwiGLU).
- **Nested GQA**: GQA backbone (n_q heads, n_kv KV groups) with prefix-sliceable Q heads.
  Uniform per-KV-group slicing: e.g. 12 Q / 3 KV groups admits `{3, 6, 9, 12}` Q heads.
  K and V stay at full size (the cache layout doesn't change with the Q-head budget).
  FFN intermediate is optionally sliced along an independent axis.
- **MatMLA**: latent attention with compressed KV cache and a separate small RoPE-K
  path. **MHA-on-decompressed-side** (paper-faithful: the paper has `n_kv == n_q`
  after decompression, not GQA). FFN intermediate is sliced along the same axis.

The training loop **samples one sub-model per step** and uses it for every microbatch
of the step. Sub-models are evaluated at each validation step so the Pareto curve
emerge directly.

## Install

```bash
pip install -e .
```

Python 3.11+, PyTorch 2.2+, MPS / CUDA / CPU.

## Run

### Local (single GPU / MPS / CPU)

```bash
# MatMLA (recommended starting point — paper-faithful nesting + compressed cache)
python train.py -config configs/matmla.yaml -log both -name matmla

# Nested GQA
python train.py -config configs/nested_gqa.yaml -log both -name nested_gqa

# Baselines (no slicing)
python train.py -config configs/base_mha.yaml -log tb -name mha
python train.py -config configs/base_gqa.yaml -log tb -name gqa

# Evaluate a trained checkpoint across all sub-models + print cache table
python evaluate.py -config configs/matmla.yaml -ckpt save/matmla/final.pt
```

`-log` is one of `wandb`, `tb`, `both`, `none`.

### SLURM cluster

```bash
# Train MatMLA on one GPU.
sbatch slurm/run.sh

# Train NestedGQA with wandb logging under a custom run name.
sbatch slurm/run.sh configs/nested_gqa.yaml -log wandb -name nested_gqa_test

# Evaluate a finished checkpoint.
sbatch slurm/eval.sh configs/matmla.yaml save/matmla/final.pt
```

See `slurm/README.md` for details on cluster-specific directives, environment
setup, and walltime budgeting.

### Quick smoke test

For a fast end-to-end run (≈10M params, wikitext-2, ~30 s on MPS, ~5 min on A100):

```bash
python train.py -config configs/smoke_real.yaml -log none -name smoke -max_steps 60
```

## Layout

```
configs/                base_mha / base_gqa / nested_gqa / matmla + smoke_real
train.py                train loop with per-step sub-model sampling
evaluate.py             sweep all sub-models + cache-size table
loss.py                 chunked cross-entropy helper (kept for reference)
matmla_cache.py         MatMLACache + cache-comparison table
slurm/
  run.sh                sbatch wrapper for `train.py`
  eval.sh               sbatch wrapper for `evaluate.py`
  README.md             cluster-specific notes + walltime budgeting
model/
  common.py             DynamicLinear, DynamicRMSNorm, RoPE, SwiGLU
  backbone.py           TransformerBlock + TransformerLM
  baseline.py           MHA + GQA (no slicing)
  nested_gqa.py         GQA with prefix-sliceable Q heads
  matmla.py             MatMLA: compress + nested decompress + RoPE-K
  factory.py            build_model, enumerate_submodels, sample_submodel, SubModelSpec
data/wikitext.py        HF datasets + tiktoken GPT-2 BPE -> memmap
utils/                  logging (wandb|tb|both|none), schedule, init, config
```

## How the slicing works

`DynamicLinear.forward_sliced(x, out_rows=k, in_cols=j)` returns
`F.linear(x[..., :j], self.weight[:k, :j], self.bias[:k] if any)`.

### NestedGQA (group-aware)

- `q_proj`, `k_proj`, `v_proj`, `o_proj` are all `DynamicLinear` of full size.
- At init, the rows of `q_proj` and the cols of `o_proj` are permuted to
  **KV-group-major** order: `[g0_h0, g1_h0, ..., g0_h1, g1_h1, ...]`.
- A sub-model with `active_q_heads` simply takes the prefix
  `q_proj.weight[:active_q_heads*head_dim]` and the matching prefix cols
  of `o_proj`. K and V stay full; only `repeat_kv`'s per-group factor changes.
- Valid `active_q_heads` are divisors of `n_q_heads` that satisfy
  `active_q_heads % n_kv_heads == 0` (uniform per-KV-group).

### MatMLA (paper-faithful, MHA-on-decompressed-side)

- Compression is always full:
  `W_DKV: d -> c_kv`, `W_DQ: d -> c_q`, `W_KR: d -> r_rope`.
- Decompression outputs `n_q * head_dim` for each of `W_UK`, `W_UV`, `W_UQ`.
  A sub-model with `active_q_heads` uses the first `active_q * head_dim`
  output cols of each. The output projection `W_O` col-slices to
  `active_q * head_dim + r_rope`. There is no GQA on the decompressed side.
- Cache layout is `(C_KV, K_RoPE)`, identical for every sub-model.

### FFN nesting (both)

`SwiGLU.forward(active_intermediate=m)` prefix-slices `gate_proj`/`up_proj`
rows and `down_proj` cols. Independent of the attention slicing axis.

## Training procedure

Each step samples ONE `SubModelSpec` (uniform over the cartesian product
of the granularity lists, plus `sample.full_prob` probability of the
absolute full sub-model). The same spec is used for every microbatch of the
step, so the gradient direction is internally consistent. Sub-model Pareto
curves are evaluated by sweeping every granularity at each validation step.

Knob: `train.sample.full_prob` in each YAML config (default `0.25`).

## Notes on the dataset

`data/wikitext.py` reads `wikitext-103-raw-v1` (or `wikitext-2-raw-v1`) via
`datasets.load_dataset` and tokenizes with the GPT-2 BPE via `tiktoken`.
The original attention_moe wikitext URL was no longer reachable, hence the
swap.
