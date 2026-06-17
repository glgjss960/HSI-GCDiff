# HSI-GCDiff

`HSI-GCDiff` implements a minimal DiffGraph-style proof protocol for unsupervised HSI pixel clustering:

```text
z_task   = task graph encoder output
z_aux    = auxiliary graph encoder output
z_target = stopgrad(z_task)
z_diff   = denoised auxiliary embedding learned from z_aux -> z_task
z_final  = normalize(z_task + alpha * z_diff)
```

The core claim is tested by comparing:

```text
z_task + alpha * z_diff
>
z_task
```

and, more importantly:

```text
z_task + alpha * z_diff
>
z_task + alpha * z_aux
```

The second comparison proves that denoising the auxiliary semantics is better than directly injecting raw auxiliary semantics.

## Stages

The default pipeline is strictly staged:

```text
build_graph
train_encoder
build_target
train_denoiser
eval
```

`all` runs these stages in order. Encoder artifacts are frozen before denoiser training. The denoiser optimizer only sees denoiser parameters.

Optional:

```text
build_teacher
```

This is reserved for future anchor/prototype target ablations. The minimal protocol does not run it.

## Environment

```bash
conda create -n hsigcdiff python=3.10 -y
conda activate hsigcdiff
pip install -r requirements.txt
cd HSI-GCDiff
```

## Smoke Test

```bash
python -m hsigcdiff.smoke_test
```

This checks that the staged v2 pipeline creates `z_task`, `z_aux`, `z_target`, `z_diff`, and injection metrics.

## Run

Full Salinas run:

```bash
python train.py --config configs/proof/salinas.json --stage all --device cuda --force
```

Manual stages:

```bash
python train.py --config configs/proof/salinas.json --stage build_graph --device cuda --force
python train.py --config configs/proof/salinas.json --stage train_encoder --device cuda --force
python train.py --config configs/proof/salinas.json --stage build_target --device cuda --force
python train.py --config configs/proof/salinas.json --stage train_denoiser --device cuda --force
python train.py --config configs/proof/salinas.json --stage eval --device cuda
```

Quick debugging:

```bash
python train.py --config configs/proof/synthetic.json --stage all --device cpu --epochs 3 --denoise-epochs 5 --force
```

Dataset commands:

```bash
python train.py --config configs/proof/salinas.json --stage all --device cuda --force
python train.py --config configs/proof/muufl.json --stage all --device cuda --force
python train.py --config configs/proof/berlin.json --stage all --device cuda --force
python train.py --config configs/proof/mdas.json --stage all --device cuda --force
```

The local workspace contains Salinas and MUUFL paths. Berlin and MDAS configs follow ETAP demo variable names and may require editing paths.

## Graph Roles

Each graph view has a role:

```json
{
  "name": "spectral_stats",
  "role": "task",
  "mode": "mean_std"
}
```

```json
{
  "name": "context_patch",
  "role": "aux",
  "mode": "center_patch"
}
```

Salinas defaults:

```text
task: spectral_stats
aux:  context_patch, spatial adjacency
```

MUUFL/Berlin/MDAS defaults:

```text
task: HSI spectral_stats
aux:  LiDAR/SAR/DSM stats, spatial adjacency
```

## Outputs

Run outputs are under `runs/<dataset>_proof_v2/`:

```text
graph.pkl
graph_meta.json
encoder/embeddings.npz
target/target.npz
denoiser/denoiser.pt
metrics.csv
config.effective.json
```

Important embeddings:

```text
z_task       frozen task embedding
z_aux        frozen auxiliary embedding
z_target     frozen stopgrad(z_task)
z_diff       sampled denoised auxiliary embedding
```

Important metric rows:

```text
z_task
z_aux
z_target
z_diff
z_task_plus_raw_aux
z_task_plus_denoised_aux
```

Important columns:

```text
alpha
sampling_steps
noise_seed
acc / oa
kappa
nmi
ari
purity
gain_vs_task
gain_vs_raw_aux_injection
```

## How To Prove The Core Idea

For each dataset, group `metrics.csv` by:

```text
method, alpha, sampling_steps
```

and average over `noise_seed`.

The core idea is supported only if there exists a practical alpha, usually `0.05`, `0.1`, or `0.2`, such that:

```text
mean ACC(z_task_plus_denoised_aux) > ACC(z_task)
```

and:

```text
mean ACC(z_task_plus_denoised_aux)
>
ACC(z_task_plus_raw_aux with the same alpha)
```

Use `gain_vs_task` for the first condition and `gain_vs_raw_aux_injection` for the second.

Report at least:

```text
ACC/OA
Kappa
NMI
ARI
Purity
mean/std over noise_seed
best alpha and sampling_steps
```

If `z_task_plus_raw_aux` is stronger than `z_task_plus_denoised_aux`, raw auxiliary fusion is enough and diffusion is not yet justified. If both are below `z_task`, the auxiliary graph or injection strength is hurting the task embedding.

## Default Scale

Salinas:

```text
n_superpixels = 2700
task_hidden_dim = 256
aux_hidden_dim = 256
latent_dim = 128
denoiser_hidden_dim = 384
diffusion_timesteps = 100
sampling_steps = 10, 20, 30
```

MUUFL/Berlin/MDAS:

```text
n_superpixels around 155-235
task_hidden_dim = 128
aux_hidden_dim = 128
latent_dim = 128
denoiser_hidden_dim = 256
diffusion_timesteps = 100
sampling_steps = 10, 20
```

These choices keep model size small for superpixel-level clustering while giving the denoiser enough capacity to map auxiliary relation semantics into the task embedding space.
