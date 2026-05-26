# HSI-GCDiff

Graph latent diffusion for unsupervised hyperspectral image pixel clustering.

This implementation is a clean research prototype built from the ideas in:

- SSGCC: HSI loading, superpixel clustering workflow, hard sample mining, pixel-level evaluation.
- DiffGraph: source relation embedding to target semantic embedding latent diffusion.
- SCD-MVC: cosine diffusion schedule, offset noise, adjacent-timestep consistency.

The training target is unsupervised HSI pixel clustering. Ground-truth labels are only used for evaluation.

## Project Layout

```text
HSI-GCDiff/
  configs/xuzhou.json        Example config
  hsigcdiff/
    data.py                  HSI .mat loading and preprocessing
    superpixels.py           SLIC superpixels, features, multi-relation graphs
    model.py                 Multi-relation GCN encoder and latent diffusion model
    diffusion.py             Diffusion schedule, denoiser, sampling
    losses.py                Reconstruction, clustering, hard contrast losses
    clustering.py            KMeans, prototypes, confidence estimates
    evaluation.py            Pixel-level clustering metrics
    trainer.py               End-to-end training loop
    smoke_test.py            Synthetic sanity test
  train.py                   CLI entry
```

## Install

```powershell
cd D:\AINet\LLM\M-A-P\HSIGD\HSI-GCDiff
pip install -r requirements.txt
```

## Run A Smoke Test

```powershell
python -m hsigcdiff.smoke_test
```

This uses a synthetic HSI cube and verifies the complete data, graph, model, diffusion, clustering, and evaluation path.

## Run On Real HSI Data

Edit `configs/xuzhou.json`:

```json
{
  "data": {
    "image_path": "../SSGCC/HSI_data/xuzhou.mat",
    "gt_path": "../SSGCC/HSI_data/xuzhou_gt.mat"
  }
}
```

Then run:

```powershell
python train.py --config configs/xuzhou.json
```

If your `.mat` file contains multiple arrays, set `image_key` and `gt_key` in the config.

## Core Pipeline

1. Standardize the HSI cube and optionally apply PCA.
2. Generate SLIC superpixels on PCA-reduced bands.
3. Build explicit multi-relation superpixel graphs:
   - spectral kNN graph
   - spatial adjacency graph
   - context/patch kNN graph
   - pixel-superpixel association for final label projection
4. Encode each graph relation with a dedicated GCN branch.
5. Fuse relation embeddings with attention.
6. Build EMA teacher and prototype-corrected target embeddings.
7. Train latent diffusion to denoise source relation embeddings toward semantic target embeddings.
8. Train clustering with DEC-style KL, view consistency, reconstruction, and hard sample contrast.
9. Cluster denoised superpixel embeddings and map labels back to pixels.

## Notes

The first version intentionally keeps dependencies simple and does not require DGL. The graph convolutions use normalized sparse adjacency matrices with `torch.sparse.mm`.

