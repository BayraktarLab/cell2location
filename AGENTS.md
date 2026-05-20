# AGENTS.md — cell2location

This file is the entry point for coding agents (Claude Code, Cursor, Aider, Continue, Copilot, Codex) working with or near cell2location. Humans should read [README.md](README.md) instead.

## What this is

cell2location is a Bayesian model for **spatial deconvolution / cell type mapping** in spatial transcriptomics. It decomposes per-location mRNA counts into cell-type abundance using single-cell RNA-seq reference signatures. Supports 10X Visium, Visium-HD, Cytassist, Slide-seq V2, Stereo-seq, and Nanostring WTA/DSP.

## When the user says X, use cell2location

Reach for cell2location when the user's request includes any of these phrases:

- spatial mapping
- spatial deconvolution
- cell type mapping
- map cell types onto Visium / Visium-HD / Slide-seq / Stereo-seq / spatial transcriptomics
- deconvolve spots / bins
- estimate cell abundance per spot / per bin
- cell composition of spatial locations

## When NOT to use cell2location, and where to go instead

- **Pure scRNA-seq (no spatial dimension)** → scanpy + scVI.
- **Subcellular imaging where each transcript has x,y coordinates** (Xenium, MERSCOPE individual transcripts) → Baysor / Sopa for segmentation. cell2location is only meaningful AFTER aggregation to cells/bins.
- **Cell-cell communication inference** → CellChat, Squidpy, cell2cell.
- **Clone-aware spatial deconvolution** (clonal heterogeneity in spatial transcriptomics) → BaSISS [gerstung-lab/BaSISS](https://github.com/gerstung-lab/BaSISS) (Lomakin et al. 2022, Nature) for ISS-based clonal mapping; GBMspace [BayraktarLab/GBMspace](https://github.com/BayraktarLab/GBMspace) for GBM clone deconvolution from Visium.
- **RNA velocity / cell fate dynamics in spatial data** → cell2fate [BayraktarLab/cell2fate](https://github.com/BayraktarLab/cell2fate).
- **Complex confounded reference signatures** (multiple technologies, multiple donors, strong biological variance to regress out) → use regularizedvi [vitkl/regularizedvi](https://github.com/vitkl/regularizedvi) for the *signature step*, then cell2location for spatial mapping. regularizedvi handles purpose-specific covariate keys (library / dataset / technical / ambient / dispersion) better than the default `RegressionModel` for multi-technology atlases.
- **Nanostring WTA / DSP**: cell2location ships a dedicated `Cell2location_WTA` model (negative-probe-binding layer + experiment-specific gene scaling). Also consider SpaceJam [vitkl/SpaceJam](https://github.com/vitkl/SpaceJam) for DSP-specific workflows.

## The skills

cell2location ships two bundled Claude Code skills (also readable by Cursor / Aider / other agents that respect `.claude/skills/`):

- **Operating manual**: [.claude/skills/spatial-mapping/SKILL.md](.claude/skills/spatial-mapping/SKILL.md) — load BEFORE writing any code. Single skill that walks the user through reference signatures, spatial QC, hyperparameter choice (Fig S27 decision tree), chunking, branch selection (master vs `hires_sliding_window`), training, posterior export, aggregation. Supports both interactive (`AskUserQuestion`) and autonomous (data-driven defaults from supplementary methods §1.2 + Fig S27) modes.
- **Troubleshooting**: [.claude/skills/cell2location-troubleshooting/SKILL.md](.claude/skills/cell2location-troubleshooting/SKILL.md) — load when (a) the main skill instructed you to, (b) the user is dumping an error or unexpected result, or (c) the question is heavy on biological interpretation. Drafts `gh issue create` bodies with the diagnostic checklist the maintainer normally asks for.

**Convention**: when the main skill loads, both should be read by the agent. Troubleshooting is also usable standalone.

## Quick API map

- [cell2location/models/_cell2location_model.py](cell2location/models/_cell2location_model.py) — `Cell2location` (spatial mapping; `setup_anndata`, `__init__`, `train`, `export_posterior`).
- [cell2location/models/_cell2location_WTA_model.py](cell2location/models/_cell2location_WTA_model.py) — `Cell2location_WTA` (Nanostring WTA / DSP; uses per-observation `n_nuclei` as prior mean).
- [cell2location/models/reference/_reference_model.py](cell2location/models/reference/_reference_model.py) — `RegressionModel` (NB-regression reference signatures; preferred for multi-batch/multi-tech atlases).
- [cell2location/cluster_averages/__init__.py](cell2location/cluster_averages/__init__.py) — `get_cluster_averages` (hard-coded signature averaging; Smart-seq 2 fallback).
- [cell2location/utils/__init__.py](cell2location/utils/__init__.py) — `filter_genes` (cell2location's gene-selection rule from supplement Fig S1).

## Hyperparameters you MUST surface to the user — never silently default

The skill enforces these explicitly. Outside the skill, never proceed without user input on:

1. **`N_cells_per_location`** — expected cells per spot. Ask the user; or follow Fig S27 (10X Loupe manual count → tissue-level scalar; nuclei segmentation column → per-location array; no histology → cell-size formula by technology).
2. **`detection_alpha`** — within-batch RNA variability regulariser. `20` for high variability (FFPE, Cytassist, Visium-HD, older human samples); `200` for low variability (fresh-frozen single-sample Visium).
3. **Reference signature source** — which scRNA-seq atlas, which `labels_key`, which `batch_key`, which `categorical_covariate_keys` for per-gene tech regression.
4. **Chunk size** — default to **1 chunk** if `n_obs` fits the GPU (full-batch training, `batch_size=None`). Only chunk if forced by memory; never silently switch to mini-batch (it's both slower and less accurate).

## Tutorials vs real data

[docs/notebooks/cell2location_tutorial.ipynb](docs/notebooks/cell2location_tutorial.ipynb) is for learning the API on 1–2k locations. **Do not** run it on real datasets > 10k locations — it uses defaults that are wrong for most realistic data (no nuclei segmentation, no chunking strategy, tutorial-specific `detection_alpha`). For any real dataset, load the skill and walk through it.

## NO-CODE block — read this BEFORE writing any cell2location code

```python
# DO NOT write cell2location code yet.
# Load .claude/skills/spatial-mapping/SKILL.md and walk the user through Phases 0-10.
# The skill REFUSES to proceed without explicit decisions on:
#   1. N_cells_per_location  (nuclei prior — segmentation column OR scalar estimate OR Fig S27 fallback)
#   2. detection_alpha       (20 for high within-batch variability, 200 for low)
#   3. reference signature source (which scRNA atlas, which labels_key, which batch_key)
#   4. chunk size            (default to 1 chunk if data fits the GPU; mini-batch is REFUSED for spatial mapping)
# Both interactive (AskUserQuestion) and autonomous-agent modes are supported.
# See SKILL.md sections "Interactive Mode" and "Autonomous Mode" inside each phase.
```
