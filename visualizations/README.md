# VECA Visualizations

This folder contains Colab/Jupyter notebooks for inspecting VECA dense features,
core attention behavior, and attention-block FLOPs. The qualitative notebooks
run a trained checkpoint on a single image; the FLOPs notebook benchmarks
standalone attention blocks.

## Notebooks

### `core_attention_layers_budgets.ipynb`

Visualizes core-to-patch attention across transformer layers and active-core
budgets.

The output grid is organized as:

```text
rows    = active core budgets, e.g. K = 8, 16, 24, 32, 48, 64
columns = layers, e.g. L2 through L12
```

For each budget, the notebook extracts query/core-to-patch attention maps,
fits a joint UMAP across selected layers, reshapes the colors back to the patch
grid, and displays the result inline.

Interpretation note: colors are comparable across layers within the same
active-core budget. They should not be interpreted as directly comparable
across different budget rows, because the attention feature dimension changes
with `K`.

### `multi_resolution_dense_features.ipynb`

Compares dense patch features across resolutions and baseline models.

The notebook runs a single image at multiple resolutions, defaulting to:

```text
256, 384, 512, 768, 1024
```

It extracts patch features, fits one joint UMAP per model across resolutions,
and displays:

- dense feature UMAP colors,
- UMAP overlays on the input image,
- global/CLS-to-patch cosine similarity maps.

The default comparison set includes:

- VECA / Ours-B, `K = 64`,
- DINOv3-B,
- DINOv2-Reg-B,
- DINOv2-B,
- ClearCLIP-B/16.

### `attention_block_flops_benchmark.ipynb`

Benchmarks the FLOPs of one VECA attention block against the corresponding
DINOv3 attention block across model sizes and image resolutions.

The benchmark isolates attention-block computation only:

```text
q/k/v projections + attention + output projection
```

## Required Inputs

The qualitative notebooks have a configuration cell near the top. Set:

```python
REPO_DIR = "/path/to/VECA"
CKPT_PATH = "/path/to/checkpoint.pt"
IMAGE_PATH = "/path/to/image.jpg"
```

The FLOPs notebook only requires:

```python
REPO_DIR = "/path/to/VECA"
```

For Colab, these can also point to mounted Google Drive paths.

## Dependencies

Install notebook dependencies in Colab:

```python
!pip -q install timm transformers umap-learn matplotlib pandas
```

For DINOv3 baselines, Hugging Face access may be gated. If needed, run:

```python
!pip -q install -U huggingface_hub

from getpass import getpass
from huggingface_hub import login

login(token=getpass("Paste Hugging Face READ token: "), add_to_git_credential=False)
```

Then request/confirm access for:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```
