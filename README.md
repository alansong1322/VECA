# Do Vision Transformers Need All-to-All Attention?

## Global Communication Through Elastic Learned Cores

Official research code for **VECA** (**V**isual **E**lastic-**C**ore
**A**ttention), a vision transformer that tests whether effective visual
representations require direct all-to-all interaction between image patches.

[Paper](https://arxiv.org/abs/2605.12491v2) | arXiv:2605.12491v2

VECA removes direct patch-to-patch attention and routes global communication
through a small set of learned **core tokens**. Crucially, it does not compress
the image into a latent bottleneck: the full set of dense, spatially aligned
patch tokens is preserved and updated in every layer. For `N` image patches and
`C` active cores, VECA uses `2NC + C^2` attention interactions instead of the
quadratic `N^2` patch-to-patch interactions of a standard Vision Transformer.
For fixed `C`, attention therefore scales linearly with image size. A single
nested-trained model can also vary `C` at inference time, providing an elastic
compute-accuracy trade-off without retraining.

## Figures

<p align="center">
  <img src="assets/teaser.png" alt="VECA teaser figure" width="900">
</p>

<p align="center">
  <img src="assets/architecture.png" alt="VECA architecture figure" width="900">
</p>

## Abstract

Vision Transformers (ViTs) achieve strong data-driven scaling by leveraging
all-to-all self-attention among patch tokens. This design implicitly assumes
that direct pairwise patch interactions are necessary for effective
representation learning, while incurring quadratic cost as image resolution
increases. VECA challenges that assumption. It uses core-periphery structured
attention mediated by a small, resolution-invariant set of learned cores:
patches exchange global information exclusively through the cores, while every
dense patch token is preserved and iteratively updated across layers. This
reduces attention complexity from `O(N^2)` to `O(N)` for a fixed core budget.

Unlike latent-token cross-attention architectures, VECA enables sparse global
communication without compressing the spatial representation itself. Nested
training along the core axis lets one model elastically trade computation for
accuracy at inference time. When distilled from a frozen DINOv3 teacher, VECA
learns representations that support both global recognition and dense
prediction, remains competitive with full-attention backbones, and outperforms
the evaluated linear-complexity alternatives on most benchmarks. Without
explicit semantic supervision, its cores also develop organized object- and
part-level roles that support label transfer across video frames.

## Contributions

- **Sparse global communication without spatial compression.** VECA removes
  direct patch-to-patch interaction while preserving and continually updating
  the full dense patch representation through a learned core-periphery
  communication graph.
- **Elastic computation from one model.** Nested training over ordered core
  prefixes allows the active budget to change at inference time, smoothly
  trading computation for accuracy without retraining.
- **Emergent semantic core structure.** Repeated interaction with the dense
  patch stream produces semantically organized cores without explicit
  supervision; these compact representations support object-label transfer
  across frames.

## Selected Findings

- In frozen-backbone evaluation, VECA-B/16 leads the evaluated
  linear-complexity architectures on six of seven classification, segmentation,
  and depth benchmarks while remaining close to the full-attention ViT-B/16.
- At `512 x 512` resolution, VECA-B/16 with `C = 8` retains `95.7%` of the
  full-attention ViT-B/16's PASCAL VOC performance while using only `1.6%` as
  many attention interactions.
- On DAVIS-2017 label propagation, 64 learned cores provide a `25.3x` token
  compression and retain `81.0%` of dense-patch `J&F`, outperforming equally
  sized uniform-sampling and k-means references.

## Method Overview

Given patch tokens `Z = {z_1, ..., z_N}` and an ordered bank of learned core
tokens `R_M = {r_1, ..., r_M}`, VECA selects an active prefix
`R_C = R_M[:C]`. Each transformer block forms the sequence `[R_C; Z]`, but uses
a block-sparse attention pattern:

```text
cores   attend to: cores + patches
patches attend to: cores only
```

Equivalently:

```text
R' = Attn(R_C, [R_C; Z], [R_C; Z])
Z' = Attn(Z,   R_C,      R_C)
```

The cores form a fully connected communication interface, while patch tokens
retain a dense per-patch representation. The resulting attention graph has
diameter 2: information from one patch can influence another through a core in
two blocks. VECA also assigns each core a learned 2D coordinate used by RoPE.
Core coordinates evolve across layers through a small coordinate prediction
head, allowing cores to develop spatially and semantically meaningful behavior.
The first active core serves as the global (`[CLS]`) representation.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- scripts/
|   |-- pretrain_object365_ddp.py
|   `-- finetune_multires_object365_ddp.py
|-- veca/
|   |-- config.py
|   |-- data.py
|   |-- model.py
|   |-- teacher.py
|   |-- train_pretrain.py
|   |-- train_multires_finetune.py
|   `-- ...
`-- visualizations/
    |-- README.md
    |-- core_attention_layers_budgets.ipynb
    |-- attention_block_flops_benchmark.ipynb
    `-- multi_resolution_dense_features.ipynb
```

## Environment Setup

VECA training is intended for a CUDA Linux environment with Python 3.10 or
newer. A typical setup is:

```bash
conda create -n veca python=3.10 -y
conda activate veca
```

Install PyTorch and TorchVision for your CUDA version following the official
PyTorch instructions. Then install this repository in editable mode:

```bash
git clone https://github.com/alansong1322/VECA.git
cd VECA
pip install -e .
```

Core training dependencies are listed in `pyproject.toml` and include PyTorch,
TorchVision, timm, Transformers, NumPy, tqdm, Weights & Biases, and `dion`.
Distributed training uses `torchrun` with PyTorch DDP.

For the visualization notebooks, install the notebook-only extras manually:

```bash
pip install umap-learn matplotlib pandas
```

## Model Families

VECA model scale is controlled by `model_family` in `veca/config.py`.

| Family | Layers | Hidden Dim | Heads | Max Cores |
| --- | ---: | ---: | ---: | ---: |
| `small` | 12 | 384 | 6 | 64 |
| `splus` | 12 | 384 | 6 | 64 |
| `base` | 12 | 768 | 12 | 64 |
| `large` | 24 | 1024 | 16 | 64 |

The default active-core budget set is:

```text
8, 16, 24, 32, 40, 48, 56, 64
```

## Data Setup

The current training scripts are configured for Object365-style image folders.
Set either explicit train/validation directories:

```bash
export OBJECT365_TRAIN_DIR=/path/to/objects365/train
export OBJECT365_VAL_DIR=/path/to/objects365/val
export OBJECT365_INDEX_DIR=./indices
```

or set a shared root:

```bash
export OBJECT365_ROOT=/path/to/objects365
export OBJECT365_INDEX_DIR=./indices
```

The first main process builds image index files automatically when they are
missing. Index paths can also be overridden directly:

```bash
export OBJECT365_TRAIN_INDEX=/path/to/object365_train_paths.txt
export OBJECT365_VAL_INDEX=/path/to/object365_val_paths.txt
```

## Checkpoints

Pretrained VECA checkpoints are available through Google Drive:

```text
https://drive.google.com/drive/folders/1MpipJtZlhcYQqTUa4AnZ5kuepUIUcNA1?usp=sharing
```

Large checkpoint files are intentionally ignored by git.

Load a checkpoint in one line:

```python
from veca import load_model

model = load_model("/path/to/checkpoint.pt", device="cuda")
```

## Training

### 1. Pretrain With Nested Core Budgets

```bash
torchrun --standalone --nproc_per_node=6 scripts/pretrain_object365_ddp.py
```

This stage trains VECA with nested active-core budgets and DINOv3 distillation
at the configured image size.

Default checkpoint names are configured in `PretrainConfig`:

```text
veca_pretrain_dinov3vitb16_256_q64_nested.pt
veca_pretrain_dinov3vitb16_256_q64_nested_best.pt
```

### 2. Multi-Resolution Finetuning

```bash
torchrun --standalone --nproc_per_node=6 scripts/finetune_multires_object365_ddp.py
```

This stage loads the pretraining checkpoint when available and trains across
multiple resolutions and active-core budgets.

Default checkpoint names are configured in `MultiResFinetuneConfig`:

```text
veca_multires_finetune_dinov3vitb16_q64_nested.pt
veca_multires_finetune_dinov3vitb16_q64_nested_best.pt
```

Checkpoints are intentionally ignored by git. Download released weights from
the checkpoint link above, or store local experiment weights outside the source
tree.

## Configuration

Main defaults live in `veca/config.py`:

- `SharedConfig`: architecture, teacher, optimizer, and dataset defaults.
- `PretrainConfig`: single-resolution pretraining settings.
- `MultiResFinetuneConfig`: multi-resolution finetuning settings.

For small experiments, edit the config dataclasses directly. For more controlled
runs, create a small Python launch script that constructs a config and calls the
corresponding training entry point.

Example:

```python
from veca.config import MultiResFinetuneConfig
from veca.train_multires_finetune import main

cfg = MultiResFinetuneConfig()
cfg.model_family = "base"
cfg.total_steps = 50000
cfg.init_ckpt = "veca_pretrain_dinov3vitb16_256_q64_nested_best.pt"
cfg.ft_ckpt_name = "veca_multires_finetune_dinov3vitb16_q64_nested.pt"

main(cfg)
```

## Visualization

See [visualizations/README.md](visualizations/README.md) for notebook details.

The current notebooks are:

- [core_attention_layers_budgets.ipynb](visualizations/core_attention_layers_budgets.ipynb):
  visualizes core-to-patch attention behavior across layers and active-core
  budgets.
- [multi_resolution_dense_features.ipynb](visualizations/multi_resolution_dense_features.ipynb):
  compares dense features across resolutions and models with joint UMAP.
- [attention_block_flops_benchmark.ipynb](visualizations/attention_block_flops_benchmark.ipynb):
  compares attention-block FLOPs against corresponding DINOv3 models.

## Notes on Normalization

VECA and the DINO baselines use standard ImageNet channel normalization
constants:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

## Citation

If you use this repository, please cite:

```bibtex
@misc{song2026vision,
  title  = {Do Vision Transformers Need All-to-All Attention? Global
            Communication Through Elastic Learned Cores},
  author = {Alan Z. Song and
            Yinjie Chen and
            Mu Nan and
            Deva Ramanan and
            Michael J. Tarr and
            Andrew F. Luo},
  year   = {2026},
  eprint = {2605.12491},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2605.12491v2}
}
```
