# HSI-GCDiff

`HSI-GCDiff` is a clean proof-of-denoising protocol for unsupervised HSI pixel clustering. It rewrites the previous joint-training prototype into frozen stages:

1. Build multi-view superpixel graphs.
2. Train and freeze a multi-view graph encoder.
3. Build and freeze an ETAP-style anchor teacher.
4. Build and freeze the diffusion target latent.
5. Train only the latent denoiser.
6. Compare ETAP/source/fuse/target/noised/denoised with one evaluator.

The implementation is meant to test whether graph latent diffusion itself improves clustering after the graph encoder and teacher are fixed.

## Code Layout

- `hsigcdiff/data.py`: `.mat` loading, PCA/standardization, synthetic data.
- `hsigcdiff/superpixels.py`: SLIC superpixels and superpixel feature aggregation.
- `hsigcdiff/graph_builder.py`: spectral/context/spatial superpixel graph construction.
- `hsigcdiff/encoder.py`: multi-view graph autoencoder and frozen `z_source/z_fuse`.
- `hsigcdiff/etap_teacher.py`: ETAP-lite anchor indicator teacher, plus external `Final_Z/G_A` loading.
- `hsigcdiff/target_builder.py`: frozen `z_target = Y_A P` from anchor assignment and graph latent prototypes.
- `hsigcdiff/diffgraph_diffusion.py`: DiffGraph-style latent denoising objective.
- `hsigcdiff/proof_protocol.py`: staged pipeline and metrics export.
- `train.py`: command-line entry point.

## Environment

```bash
conda create -n hsigcdiff python=3.10 -y
conda activate hsigcdiff
pip install -r requirements.txt
```

Run commands from this directory:

```bash
cd HSI-GCDiff
```

## Smoke Test

```bash
python -m hsigcdiff.smoke_test
```

This only verifies that the full pipeline runs. It does not validate HSI performance.

## Data Fields

Each dataset config uses MATLAB arrays. Check keys with:

```bash
python -c "from hsigcdiff.data import mat_keys; print(mat_keys('../Salines/Salinas_corrected.mat'))"
```

Salinas uses:

- image: `../Salines/Salinas_corrected.mat`, key `salinas_corrected`
- GT: `../Salines/Salinas_gt.mat`, key `salinas_gt`

MUUFL uses:

- `../ETAP/data/MUUFL/HSI.mat`, key `HSI`
- `../ETAP/data/MUUFL/LiDAR.mat`, key `LiDAR`
- `../ETAP/data/MUUFL/gt.mat`, key `gt`

Berlin follows the ETAP demo convention:

- `data_HS_LR.mat`, key `data_HS_LR`
- `data_SAR_HR.mat`, key `data_SAR_HR`
- `TestImage.mat + TrainImage.mat` as GT

MDAS follows the ETAP demo convention:

- `MDAS-Sub1-HSI.mat`, key `Data_HSI`
- `MDAS-Sub1-DSM.mat`, key `Data_DSM`
- `MDAS-Sub1-GT.mat`, key `GT`

The local workspace currently contains Salinas and MUUFL. Berlin and MDAS configs are templates; put the official files under the configured paths or edit the JSON paths.

## Run A Dataset

Full pipeline:

```bash
python train.py --config configs/proof/salinas.json --stage all --device cuda
```

Run stages manually:

```bash
python train.py --config configs/proof/salinas.json --stage build_graph --device cuda --force
python train.py --config configs/proof/salinas.json --stage train_encoder --device cuda --force
python train.py --config configs/proof/salinas.json --stage build_teacher --device cuda --force
python train.py --config configs/proof/salinas.json --stage build_target --device cuda --force
python train.py --config configs/proof/salinas.json --stage train_denoiser --device cuda --force
python train.py --config configs/proof/salinas.json --stage eval --device cuda
```

Useful quick overrides:

```bash
python train.py --config configs/proof/salinas.json --stage all --device cuda --epochs 50 --denoise-epochs 200 --force
```

Outputs are written to `runs/<dataset>_proof/`:

- `graph.pkl`, `graph_meta.json`
- `encoder/embeddings.npz`
- `teacher/teacher.npz`
- `target/target.npz`
- `denoiser/denoiser.pt`
- `metrics.csv`
- `config.effective.json`

## How To Read `metrics.csv`

Important rows:

- `ETAP_hard`: hard anchor-propagated teacher clustering.
- `Y_anchor`: soft anchor assignment argmax.
- `z_source`: mean of per-view graph encoder latents.
- `z_fuse`: frozen graph latent fusion.
- `z_target`: frozen projected anchor-centroid target.
- `z_noised`: `q(z_source, t)` before denoising.
- `z_denoised`: denoiser prediction from the same noisy latent and timestep.

Metrics:

- `acc`/`oa`: clustering accuracy after Hungarian matching.
- `kappa`: Cohen kappa after matching.
- `nmi`, `ari`: label-invariant clustering metrics.
- `purity`: cluster purity.
- `gain_vs_source`, `gain_vs_fuse`: direct gain over frozen encoder baselines.

The strongest proof is not just `z_denoised > z_source`. Check:

1. For the same `t` and `noise_seed`, `z_denoised` should outperform `z_noised`.
2. Averaged over noise seeds, `z_denoised` should outperform `z_source` and preferably `z_fuse`.
3. `z_target` should be competitive; otherwise the teacher target is too weak.
4. The denoiser loss should decrease while encoder/teacher/target artifacts stay unchanged.

## Dataset Commands

```bash
python train.py --config configs/proof/salinas.json --stage all --device cuda
python train.py --config configs/proof/muufl.json --stage all --device cuda
python train.py --config configs/proof/berlin.json --stage all --device cuda
python train.py --config configs/proof/mdas.json --stage all --device cuda
```

## Default Scale

The defaults follow the scale used by the referenced HSI work:

- Salinas: 2700 superpixels, 128-d latent, 256 encoder hidden, 384 denoiser hidden.
- MUUFL: 200 superpixels/anchors, 128-d latent, 128 encoder hidden, 256 denoiser hidden.
- Berlin: 155 superpixels/anchors, 128-d latent, 128 encoder hidden, 256 denoiser hidden.
- MDAS: 235 superpixels/anchors, 128-d latent, 128 encoder hidden, 256 denoiser hidden.

Use `fusion="mean"` for the first proof run. Attention or concat fusion can be added as an ablation after the denoising effect is established.

## External ETAP Teacher

The default `teacher.backend="python_etap_lite"` is an ETAP-inspired Python implementation. To use an exported official ETAP teacher, create an `.npz` with:

- `Final_Z`: node-to-anchor assignment.
- `G_A`: anchor-to-cluster indicator, or `anchor_labels`.

Then set:

```json
"teacher": {
  "backend": "load_npz",
  "path": "path/to/etap_teacher.npz"
}
```
