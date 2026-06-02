---
name: spatial-mapping
description: "Run cell2location on real spatial transcriptomics data (Visium / Visium-HD / Cytassist / Slide-seq / Stereo-seq / Nanostring WTA). TRIGGER when the user asks to spatially map / deconvolve cell types onto spatial transcriptomics, or to estimate cell type abundance per spot/bin. Forces explicit decisions on N_cells_per_location, detection_alpha, reference signature source, and chunking. Supports both interactive (AskUserQuestion) and autonomous (data-driven defaults from supplementary methods Fig S27 + §1.2) modes. Generates parametrised notebooks + LSF/Slurm/local launchers."
user-invocable: true
---

# spatial-mapping — apply cell2location to a real spatial transcriptomics dataset

This skill is the operating manual for running cell2location on the user's own data. It is **single-skill, format-plan-style** (instructions + `<reference>` tag), **dual-mode** (interactive `AskUserQuestion` / autonomous data-driven), and it **forces decisions** the maintainer (vitkl) routinely answers on the issue tracker.

Companion skills (when this skill is loaded, also load both):

- [cell2location-context/SKILL.md](../cell2location-context/SKILL.md) — owns the persistent `SPATIAL_MAPPING_CONTEXT.md` (project goals, reference, target populations, success/failure criteria + technical decisions). Invoked automatically by this skill at **Phase 0a** (`--science`) and **Phase 8.5** (`--technical`). Standalone-invocable via `/cell2location-context` to update goals between runs.
- [cell2location-troubleshooting/SKILL.md](../cell2location-troubleshooting/SKILL.md) — consults the same issue corpus AND the `SPATIAL_MAPPING_CONTEXT.md` (especially the failure criteria + technical decisions) to match symptoms; helps file `gh issue create` drafts when this skill cannot resolve a problem.

## How this skill works

You walk the user through ten phases, in order. Each phase has TWO branches:

- **`<interactive>`** — when the user is present, ask via `AskUserQuestion`.
- **`<autonomous>`** — when the user is NOT available (background agent, headless SDK run, biorxiv-style autonomous research agent), inspect the data + tissue + technology and converge on a defaulted value using the rules from [Fig S27](reference/fig_S27_hyperparameters.png) and the supplement [§1.2 hyperparameter rules](reference/hyperparameters_extract.md).

**Hard rule for autonomous mode**: NEVER refuse to proceed for lack of a hyperparameter. Always converge on an informed default and emit a markdown cell into the generated notebook documenting the assumption ("Assumed `detection_alpha=20` because 90/10 percentile ratio of total_counts is 14× > 10×; override if your slides are fresh-frozen single-batch.").

**Hard rule for both modes**: NEVER silently default. In interactive mode, ask. In autonomous mode, document the choice and the inference rule that produced it.

The output of the skill is a **set of generated artifacts**: customised parameter values for the [template notebooks](templates/), a launcher invocation, and a markdown explanation of what was chosen and why. The user can either submit the job via papermill OR open the notebook in Jupyter for interactive use.

---

# Phase 0a — Scientific-scope interview (FIRST STEP)

## Goal
Capture or load the user's scientific scope **before any technical decision is made**: why are they running cell2location, what reference, which target populations, what counts as success, what must not happen. All downstream phases consult this context.

## Action
Invoke the [cell2location-context](../cell2location-context/SKILL.md) skill with `--science`:

```
/cell2location-context --science
```

(Or call its `SKILL.md` directly when the slash-command channel is not available — read the file, follow its instructions.)

The context skill will:

1. **Auto-discover** `SPATIAL_MAPPING_CONTEXT.md` (candidate paths: `$CWD/`, `$CWD/.claude/`, `~/.claude/plans/SPATIAL_MAPPING_CONTEXT_<dataset>.md`).
2. **If found**: show the user a summary of the `## Scientific scope` block and ask: **Use** the existing context / **Re-interview** to update / **Skip** for this run.
3. **If not found**: ask the user: **Run the interview now (recommended)** _"Answering these questions can substantially improve the results from this analysis, get a more useful single-cell reference and spatial map, and lead to new discoveries."_ / **Point me at an existing handoff document** (imports its content as free-form scope) / **Skip — proceed with defaults**.
4. **Return** either a file path or the string `"skipped"`.

## How to consume the result

- **If a path was returned**: read the file's `## Scientific scope` block. Carry it forward as context for every subsequent phase. In each phase that has a scope-relevant rule (Phase 1 granularity, Phase 3 N̂ choice, Phase 4 detection_alpha, Phase 6 branch), explicitly cite the scope entry that argued for the chosen value.
- **If `"skipped"` was returned**: continue, but emit a markdown cell into the generated step1 and step2 notebooks:
  ```markdown
  # ⚠️ No SPATIAL_MAPPING_CONTEXT.md captured
  This run proceeded with autonomous defaults because the user skipped the scientific-
  scope interview. Hyperparameter choices, warnings, and refusals were not grounded in
  user-declared success/failure criteria. To improve future runs, invoke
  `/cell2location-context --science` and re-launch the workflow.
  ```

## Autonomous mode
If no `AskUserQuestion` channel is exposed, the context skill auto-skips both branches and returns `"skipped"`. This phase records the skip in the notebook (markdown cell above) and proceeds.

## Output
Either the loaded `## Scientific scope` block (passed forward as context to all phases) or a recorded skip.

---

# Phase 0 — Mode + data-discovery branch

## Goal
Determine (a) whether the user is available for questions, and (b) whether they already have their own data or need to start from a published demo.

## Interactive
Ask:
- **Mode**: am I running in an interactive session with you present (you'll see my AskUserQuestion prompts)? Or am I running autonomously (no user available, agent decides)?
- **Data status**: (a) "I have my own spatial transcriptomics data" → continue to Phase 1; (b) "I want to test the skill on published data" → run the demo path; (c) "I don't have spatial data yet" → emit data-acquisition guidance.

## Autonomous
Default to autonomous mode if the skill is invoked without an interactive `AskUserQuestion` channel (e.g. via the Claude Agent SDK without a `permission_mode='askUser'`-equivalent setting). Heuristic: if no `AskUserQuestion` tool is exposed in the current tool list, treat as autonomous.

For data discovery: look for `*.h5ad` files in the workspace root and any `data/` subdirectory. If found → assume "I have my own data". If not → run the demo path with [templates/data/download_mouse_brain.py](templates/data/download_mouse_brain.py).

## Demo path
For "I want to test" or "no data yet" cases, run:
```python
python templates/data/download_mouse_brain.py --output-dir templates/data/
```
This downloads the published 5-Visium + 6-8-snRNA mouse-brain dataset from [vitkl/cell2location_paper](https://github.com/vitkl/cell2location_paper). Validates the user's environment and gives them a published-comparable reference.

## Data-acquisition guidance (only emitted on "no data yet" path)
- For 10X Visium: 10–20 samples typically sufficient for joint modelling; use the maximum samples that share biological/technical conditions of interest.
- For Visium-HD: choose bin size (2/8/16 μm) based on tissue cell density.
- Reference scRNA-seq: same tissue ideally; 40k+ cells with 20–50 well-characterised cell types.

---

# Phase 1 — Reference signatures (step 1 notebook)

## Goal
Set the parameters for the reference-signature `RegressionModel` workflow. Output is a `signatures.csv` (genes × cell_types in linear count scale, batch-corrected) consumed by Phase 7's `Cell2location` model.

Read [reference/hyperparameters_extract.md §2](reference/hyperparameters_extract.md) for the reference-estimation method choice (NB regression vs cluster-averages). The skill defaults to `RegressionModel` (NB regression).

## Interactive
AskUserQuestion fields:
- `ref_h5ad_path` — path to scRNA reference AnnData.
- `labels_key` — column in `adata.obs` for cell-type labels.
- `batch_key` — column for batch (e.g. `sample` or `sample_id`).
- `categorical_covariate_keys` — list of extra categorical covariate columns (e.g. `['10x_kit', 'donor']`). These drive per-gene tech regression `detection_tech_gene_tg`. **Use this for multi-technology references** (10X v2 + v3 + Smart-seq, etc.); do NOT lump them all into `batch_key`. See [_reference_module.py:178](../../../cell2location/models/reference/_reference_module.py#L178).
- `continuous_covariate_keys` — list of continuous covariates (rare).
- `gene_filter_cell_count_cutoff` — default 15.
- `gene_filter_cell_percentage_cutoff2` — default 0.03 (Fig S1 rule).
- `gene_filter_nonz_mean_cutoff` — default 1.12 (Fig S1 rule).
- `max_epochs` — default formula `min(round(20000/n_cells * 400), 400)`. Surface this number.

## Autonomous
Inspect `adata_ref.obs.columns` and `adata_ref.obs.dtypes`:
1. `labels_key` candidates: match against `cell_type`, `cluster`, `annotation`, `celltype`, `cell_type_lvl*`, `leiden`, `louvain`. Pick the first column whose unique-value count is in `[10, 200]` (cell-type cardinality).
2. `batch_key` candidates: `sample`, `sample_id`, `donor`, `batch`. Pick the first present.
3. Multi-tech detection: if any obs column has values matching `10x_v2`/`10x_v3`/`smart.?seq.?2?`/`smart_seq2` → add that column to `categorical_covariate_keys`. Also add `donor` if distinct from `batch_key` selection.
4. Use defaults for gene filters and `max_epochs`.
5. Emit a notebook markdown cell: `# Auto-detected: labels_key='X' (N=...), batch_key='Y' (N=...). Override if incorrect.`

## Failure-mode warnings (apply to both modes)

- **Reference too coarse**: if `n_unique(labels_key) < 10`, warn user: "Reference has only N cell types. cell2location works best with 20–50 well-characterised types. With too few, all types appear everywhere ('3-fingered glove on a 5-fingered hand' — see issue [#395](https://github.com/BayraktarLab/cell2location/issues/395))."
- **Reference too small**: if `min_cells_per_type < 40`, warn: "Cell types with <40 cells will have noisy signatures."

## Output
Customised parameter values for [templates/step1_reference_signatures.ipynb](templates/step1_reference_signatures.ipynb).

---

# Phase 2 — Spatial data inspection + tissue inference + QC

## Goal
Understand what spatial dataset the user has (technology, sample count, total-counts distribution), filter low-quality spots, decide gene filters.

## Interactive
AskUserQuestion:
- `spatial_h5ad_path` — path to spatial AnnData.
- `spatial_technology` — `visium` / `visium-hd` / `cytassist` / `slide-seq-v2` / `stereo-seq` / `nanostring-wta`.
- `batch_key` — sample column in `adata.obs` (typically `sample` or `sample_id`).
- `total_counts_min` — default 1000 (or `np.median(total_counts) / 3` if median is lower).
- `total_counts_max` — default 200000.
- `sample_fraction_threshold` — default 0.7 (drop samples where <70% of locations pass QC).
- Gene-filter thresholds for `filter_genes`: `cell_count_cutoff=15`, `cell_percentage_cutoff2=0.15`, `nonz_mean_cutoff=1.11`.

## Autonomous
Inspect `adata.uns`, `adata.obsm`, `adata.obs`:
1. **Technology detection**:
   - `adata.uns['spatial']` present with `library_id` keys → 10X Visium / Cytassist / Visium-HD. Bin size from `scalefactors_json`.
   - No `adata.uns['spatial']` but `adata.obsm['spatial']` present → likely Slide-seq / Stereo-seq / general spatial. Inspect spot density: <10 μm spacing → Slide-seq V2 / Stereo-seq; ~55 μm → Visium-like.
   - `neg_probes` in `adata.obs` columns or `obsm` → Nanostring WTA/DSP → **switch to `Cell2location_WTA` model** (see Phase 7 branch). See [_cell2location_WTA_module.py:247](../../../cell2location/models/_cell2location_WTA_module.py#L247).
2. **Total-counts variability**: compute 90/10 percentile ratio per `batch_key`-group. Store for Phase 4.
3. **Tissue inference** (inform Phase 3 cell-size fallback): scan cell-type labels (if reference passed alongside) or look for `tissue` column in `adata.uns`.
4. Apply default QC thresholds; emit markdown cell documenting them.

## Output
Customised QC parameter values for [templates/step2_spatial_mapping.ipynb](templates/step2_spatial_mapping.ipynb).

---

# Phase 3 — `N_cells_per_location` (THE MOST IMPORTANT DECISION)

## Goal
Set `N̂` per the [Fig S27 decision tree](reference/fig_S27_hyperparameters.png). This drives absolute cell abundance estimation; getting it wrong invalidates all downstream comparisons.

Read [reference/hyperparameters_extract.md §1.2 item 1](reference/hyperparameters_extract.md) before this phase.

## Decision tree (Fig S27)

```
Q1: Paired histology / DAPI image for the same tissue section?
├── YES:
│   └── Q2: Per-location nuclei segmentation in adata.obs (e.g. `n_cell`, `nuclei`, `occupancy`)?
│       ├── YES (ADVANCED) → per-location N̂_s = occupancy × N_nuclei × scaling.
│       │   * Formula from the embryo workflow: n_cell_occupancy = occupancy * np.quantile(n_cell, 0.99999).
│       │   * v_n = 10 (high confidence).
│       │   * EFFECTIVENESS: only deterministic in hires_sliding_window branch
│       │     with use_proportion_factorisation_prior_on_w_sf=True AND
│       │     use_n_s_cells_per_location_limit=True. See Phase 6.
│       │   * Why occupancy × N × scaling > N alone: occupancy accounts for
│       │     the fraction of the spot covered by tissue, so per-location N̂_s
│       │     is correctly attenuated for partly-empty spots when used as
│       │     the hard multiplier in hires.
│       └── NO → manual count in 10X Loupe browser, 10-20 spots → single tissue-level N̂, v_n = 1
├── NO → Q3: Same-tissue histology (not paired) available?
│       ├── YES → same manual-count procedure on similar tissue
│       └── NO → Fig S27 fallback: cell-size + capture-size formula:
│                * 10X Visium (55 μm)     → N̂ ≈ 5
│                * Cytassist Visium       → N̂ ≈ 5 (low confidence)
│                * Visium-HD              → scale by bin area (2/8/16 μm bin)
│                * Slide-Seq V2 (10 μm)   → N̂ ≈ 1
│                * Stereo-seq             → N̂ ≈ 1 (bin-dependent)
│                * Nanostring WTA/DSP     → use per-region segmentation if available;
│                                          else WTA model handles via n_nuclei input
```

## Interactive
Walk the user through Q1 → Q2 → Q3 via sequential AskUserQuestion blocks. Default to "NO histology" if user doesn't engage.

## Autonomous
1. Scan `adata.obs.columns` for nuclei segmentation columns: `n_cell`, `nuclei`, `nuclei_count`, `occupancy`, `n_cell_occupancy`, `cell_count`, `n_nuclei`.
2. If **`n_cell` AND `occupancy` both present**: compute `n_cell_occupancy = adata.obs['occupancy'] * np.quantile(adata.obs['n_cell'], 0.99999)` → set per-location N̂_s; `v_n = 10`; FLAG: user needs hires branch (Phase 6).
3. If **only `n_cell`** (no occupancy): use per-location `n_cell` as N̂_s; `v_n = 10`; FLAG: occupancy scaling missing → less effective than the occupancy×N formula. User should regenerate segmentation with occupancy.
4. If **no segmentation columns**: use Fig S27 cell-size fallback by detected technology (Phase 2):
   - Visium → 5; Cytassist → 5; Visium-HD → adjust by bin area; Slide-seq V2 → 1; Stereo-seq → 1.
   - `v_n = 1`.
5. Emit notebook markdown cell documenting the chosen path: `## N_cells_per_location decision\n\nUsed Fig S27 fallback (no nuclei segmentation columns found in adata.obs).\nTechnology detected: Visium → N̂ = 5, v_n = 1.\n\nOverride: provide adata.obs['n_cell_occupancy'] column with per-location nuclei counts, AND install cell2location from the hires_sliding_window branch (Phase 6) to make this constraint deterministic.`

## Output
Customised `N_cells_per_location_column` OR `N_cells_per_location_scalar` value + `N_cells_per_location_alpha_prior` (1 or 1000) for [templates/step2_spatial_mapping.ipynb](templates/step2_spatial_mapping.ipynb).

---

# Phase 4 — `detection_alpha` (Fig S27 lower flow)

## Goal
Set the regularisation strength of the per-location detection prior `y_s ~ Gamma(α^y, α^y / y_e)`.

Read [reference/hyperparameters_extract.md §1.2 item 2](reference/hyperparameters_extract.md).

## Decision rule

```
Strong within-batch RNA-count variation NOT explained by tissue containing more cells?
├── YES → detection_alpha = 20  (less strict; common for FFPE, Cytassist, Visium-HD, older human)
└── NO  → detection_alpha = 200 (strict; fresh-frozen single-sample Visium)
```

Tissue caveat: tissues with intrinsic regions of high cell density (hippocampus, gut lymphoid follicle) explain higher RNA counts by cell density, not by detection variability → `200` may still be appropriate. The model regulariser interprets `y_s` variation as TECHNICAL noise; if it's biological-density-driven, the absolute abundance prior (Phase 3) handles it.

## Interactive
Show the user a quick histogram of `total_counts` per location, grouped by sample. Ask: "Does within-sample RNA variability look mostly technical (gradients, artifacts) or mostly biological (denser tissue regions)?"

## Autonomous
1. Compute per-sample 90/10 percentile ratio of `adata.obs['total_counts']`.
2. If ratio > 10× for any sample → `detection_alpha = 20`.
3. Technology override: FFPE / Cytassist / Visium-HD → `20` regardless.
4. Else → `detection_alpha = 200`.
5. Emit notebook markdown cell with the ratio and the choice.

## Per-batch override
If user has multiple samples with different qualities (some fresh-frozen, some FFPE), the model accepts a per-batch dict: `detection_alpha = {'sample_a': 20, 'sample_b': 200, ...}`. Both modes should surface this option when a mixed batch is detected.

## Output
Customised `detection_alpha` (scalar or dict) for [templates/step2_spatial_mapping.ipynb](templates/step2_spatial_mapping.ipynb).

---

# Phase 5 — Chunking strategy (anti-default rule)

## Goal
Decide whether to train on all data in one chunk, or split into multiple chunks. **Default is 1 chunk if data fits the GPU.** Never silently default to multiple chunks.

Per the user's brief: "not to split data in 5 chunks but to select chunk size that maximises GPU memory use - for many datasets it will be one chunk".

## Sizing formula

```
chunk_size_max ≈ available_gpu_memory_bytes / (n_vars × 8 bytes/float64 × overhead_factor)
                                                                            (overhead_factor ≈ 3-5)
n_chunks = ceil(n_obs / chunk_size_max)
```

## Interactive
AskUserQuestion:
- `gpu_memory_gb` — 40 / 80 / 96 / other.

Then compute and SHOW the user the result before generating code:
> "Your 312,847-location dataset on an 80GB A100 needs **4 chunks of ~78k each**. Each chunk will train for ~2h via stratified random allocation across your 47 samples. Confirm or override to (1) increase chunks for safety / (2) request more GPU memory."

## Autonomous
1. Default GPU memory budget: 80 GB (A100/H100).
2. Compute `chunk_size_max` from formula above with overhead_factor = 4.
3. If `n_obs ≤ chunk_size_max`: `n_chunks = 1`, `batch_size = None` (full-batch).
4. Else: `n_chunks = ceil(n_obs / chunk_size_max)`; stratified random allocation across samples (embryo workflow pattern — each chunk sees ALL samples). Emit markdown cell with the chunk plan.

## REFUSALS
- **Mini-batch training for spatial mapping (`batch_size != None`)** — REFUSE. Cite [issue #356](https://github.com/BayraktarLab/cell2location/issues/356) + supplement §1.3. Mini-batch is dramatically less accurate AND slower in wall-clock (10+ sec/iter → days/weeks vs full-batch ~1–2h per chunk on A100).

## Output
- `n_chunks`, `training_batch` (0..n_chunks-1, set per launcher invocation), chunk-assignment logic in [templates/step2_spatial_mapping.ipynb](templates/step2_spatial_mapping.ipynb).

---

# Phase 6 — Branch selection (master vs hires_sliding_window vs WTA)

## Goal
Pick the right cell2location model variant. The choice affects WHETHER `N_cells_per_location` actually constrains the model deterministically (hires) or only as a soft prior (master).

## Decision

```
Per-location nuclei segmentation provided (Phase 3 chose per-location N̂_s)?
├── YES → install hires_sliding_window branch + use the proportion-factorisation flags
└── NO  → use master branch (default)

Nanostring WTA / DSP data?
└── YES → use Cell2location_WTA (master branch already provides it; takes n_nuclei as forward arg)
```

## Why this matters — model-mechanism explanation

In **master** ([_cell2location_module.py:275-281](../../../cell2location/models/_cell2location_module.py#L275)):

```python
n_s_cells_per_location ~ Gamma(N_cells_per_location * N_cells_mean_var_ratio,
                               N_cells_mean_var_ratio)
```

This is a **soft Gamma prior**. The variable `n_s_cells_per_location` enters factorization through the `z_sr` rate `n_s_cells_per_location / b_s_groups_per_location`, but `w_sf` can drift away from it. Per-location segmentation counts as soft guidance.

In **hires_sliding_window** (with `use_proportion_factorisation_prior_on_w_sf=True` AND `use_n_s_cells_per_location_limit=True`):

```python
w_sf ~ Gamma(w_sf_mu * mean_var_ratio, mean_var_ratio)   # samples proportions
w_sf = w_sf / w_sf.sum(dim=-1, keepdim=True)             # normalize to proportions
w_sf = w_sf * n_s_cells_per_location                     # multiply by N̂_s
pyro.deterministic("w_sf", w_sf)                         # deterministic output
```

`n_s_cells_per_location` is now a **deterministic multiplier** on the final cell abundance. Per-location segmentation directly scales total cell abundance per location.

**The 2022 paper used master-style wiring and reported limited benefit from segmentation.** The hires rewiring is what makes segmentation actually effective. The `occupancy × N_nuclei × scaling` (`n_cell_occupancy`) formula from the embryo workflow further corrects for partly-empty spots by attenuating the multiplier.

## Action

If Phase 3 chose per-location N̂_s:
```
Instruct the user (interactive) OR auto-install (autonomous):
    pip install --force-reinstall git+https://github.com/BayraktarLab/cell2location.git@hires_sliding_window
Set in step2 parameters:
    use_proportion_factorisation_prior_on_w_sf = True
    use_n_s_cells_per_location_limit = True
    N_cells_per_location_alpha_prior = 1000.0
```

Else (master is fine):
```
pip install cell2location          # standard PyPI install
Set:
    use_proportion_factorisation_prior_on_w_sf = False  (or leave unset on master)
    use_n_s_cells_per_location_limit = False
```

For Nanostring WTA/DSP:
```
Use cell2location.models.Cell2location_WTA(adata, ...)
Pass n_nuclei as a per-observation column via setup_anndata's "nuclei" key.
```

## Output
Branch install command + step2 parameter values reflecting branch features.

---

# Phase 7 — Model hyperparameters (supplement §1.2 defaults)

## Goal
Set the remaining model hyperparameters. All have defaults from supplement §1.2; users rarely override.

Read [reference/hyperparameters_extract.md "Other priors"](reference/hyperparameters_extract.md).

| Hyperparameter | Default | When to override |
|---|---|---|
| `n_groups` (R) | 50 | Almost never |
| `A_factors_per_location` (Â) | 7 | Almost never |
| `B_groups_per_location` (B̂) | 7 | Almost never |
| `w_sf_mean_var_ratio` (v^w) | 5 | Almost never |
| `N_cells_mean_var_ratio` (v^n) | 1 (global N̂) / 10 (per-location N̂_s) | Per Phase 3 |
| `N_cells_per_location_alpha_prior` | None (master) / 1000.0 (hires) | Per Phase 6 |
| `use_per_cell_type_normalisation` | False | Advanced; per-CT library-size adjustment |
| `detection_hyp_prior['mean_alpha']` | 10 | Almost never |

## Auto-derived hyperpriors (always computed by the template; not user-set)

Per supplement §1.2 item 3 + embryo step2 cells 36-39:

```python
expected_y_e = (adata_vis.obs[['sample', 'total_counts']].groupby('sample').mean()
                / (inf_aver.sum(0) * np.mean(N_cells_per_location)).mean())
mean_alpha_prior = np.round(((expected_y_e.mean() ** 2) / expected_y_e.var()).values[0] / 3, 2)
detection_cell_type_prior_alpha = np.round(((inf_aver.sum(0).mean() ** 2)
                                             / inf_aver.sum(0).var()) * 20, 2)
```

These are passed as `detection_hyp_prior={'mean_alpha': mean_alpha_prior}` and `detection_cell_type_prior_alpha=detection_cell_type_prior_alpha` to the `Cell2location` constructor.

## Interactive
Surface the defaults table; ask if any override.

## Autonomous
Use defaults verbatim. Emit a markdown cell listing the chosen values.

## Output
Hyperparameter values for [templates/step2_spatial_mapping.ipynb](templates/step2_spatial_mapping.ipynb).

---

# Phase 8 — Training + posterior export (supplement §1.3)

## Goal
Set `max_epochs`, posterior-export options. Both have data-size-dependent defaults.

## `max_epochs` tier

| Dataset size | `max_epochs` |
|---|---|
| Small (<5k locations) | 5,000 – 10,000 |
| Medium (10k – 40k) | 20,000 – 30,000 |
| Large (>100k, chunked) | 30,000 – 80,000 per chunk |

ELBO oscillations late in training are normal and OK — do not stop on first oscillation. Train to full `max_epochs`. [Issue #327](https://github.com/BayraktarLab/cell2location/issues/327).

## `train()` defaults (verbatim from supplement §1.3)

```python
mod.train(
    max_epochs=max_epochs,
    batch_size=None,           # FULL BATCH — never override
    lr=0.002,                  # ADAM, fixed per supplement
    train_size=1,              # use all data
    accelerator='gpu',
)
```

## Posterior export (supplement §1.3 + issues #278/#360)

```python
adata_vis = mod.export_posterior(
    adata_vis,
    use_quantiles=True,        # MANDATORY for n_obs > 100k; default-on otherwise
    add_to_obsm=['means', 'q05', 'q50', 'q95'],
    sample_kwargs={
        'batch_size': int(np.ceil(adata_vis.n_obs / 4)),
        'accelerator': 'gpu',
        'return_observed': False,
    },
    exclude_vars=['data_target'],
)
```

The paper uses `q05` in all figures (slightly more accurate than mean for absolute abundance).

## QC after training

1. `mod.plot_history(5000)` — ELBO curve. Skip first 5k for visibility.
2. `mod.plot_QC(summary_name='q05')` — posterior predictive log-log plot (per supplement §1.3).
3. Spatial plot of `y_s` — should resemble total RNA count distribution.

## Interactive
Surface the tier-chosen `max_epochs`; ask if user wants to override.

## Autonomous
Use the tier from Phase 5's `n_chunks` × `chunk_size`. Emit markdown cell with the chosen value and the rationale ("`n_obs=78k`, medium tier → `max_epochs=30000`").

## REFUSALS

- `batch_size != None` → REFUSE (already in Phase 5).
- `num_samples=1000` on `n_obs > 100k` WITHOUT `use_quantiles=True` → REFUSE. OOM. [Issue #278](https://github.com/BayraktarLab/cell2location/issues/278).
- Log-transforming the spatial counts before model fitting → REFUSE. Breaks NB likelihood. [Issue #386](https://github.com/BayraktarLab/cell2location/issues/386).

## Output
`max_epochs`, posterior-export sample_kwargs for [templates/step2_spatial_mapping.ipynb](templates/step2_spatial_mapping.ipynb).

---

# Phase 8.5 — Implementation-completeness check (BEFORE LAUNCH)

## Goal
Sweep every technical decision made in Phases 1–8, confirm all slots are filled, persist them into `SPATIAL_MAPPING_CONTEXT.md`, and **cross-check them against the user's `## Scientific scope`** (especially failure criteria). Block the launch in Phase 9 if any cross-check fires a hard violation that the user has not acknowledged.

## Action
Invoke the [cell2location-context](../cell2location-context/SKILL.md) skill with `--technical`:

```
/cell2location-context --technical
```

Pass it the current notebook parameter dict (or, if running through the formal workflow-state markdown, the per-phase summary the skill has been emitting).

The context skill will:

1. **Sweep each slot** in the Phase 1–8 decision table (see [reference/technical_completeness_rubric.md](../cell2location-context/reference/technical_completeness_rubric.md)). Fill any EMPTY slot with the defensible default OR ask the user when no default is sensible.
2. **Cross-check** the chosen decisions against the user's failure criteria. Example fires:
   - Failure criterion "abundance varies 10× across visually similar regions" + `detection_alpha=200` chosen → flag.
   - Failure criterion "subtype mixing" + per-location segmentation hinted in scope + Phase 6 chose `master` → flag.
   - Target population lumped in chosen `labels_key` → flag, route to issue #395.
3. **Persist** the `## Technical decisions` and `## Outstanding gaps` blocks.
4. **Return** the file path AND a boolean `safe_to_launch`. If `False`, ask the user: "N cross-checks flagged. Launch anyway / fix and re-run / abort?" Default to **fix and re-run**.

## Autonomous mode
Cross-checks still run; defaults are applied silently. `safe_to_launch=False` becomes a warning markdown cell in the launcher invocation log; the workflow proceeds (autonomous runs cannot block on user input).

## Output
Updated `SPATIAL_MAPPING_CONTEXT.md` with the technical decisions; a go/no-go signal for Phase 9.

---

# Phase 9 — Compute infrastructure (launch)

## Goal
Submit the job. Pick the right launcher for the user's compute.

## Choices

- **LSF cluster (Sanger, Crick LSF)** → [templates/bsub.sh](templates/bsub.sh)
- **Slurm cluster (Crick Slurm, NIH, cloud)** → [templates/sbatch.sh](templates/sbatch.sh)
- **Single local GPU (laptop / workstation)** → [templates/run_local.sh](templates/run_local.sh)
- **Interactive Jupyter** (user wants to step through cells manually) → open `templates/step2_spatial_mapping.ipynb` in Jupyter Lab after setting the parameters cell at top.

## Interactive
Ask which compute the user has.

## Autonomous
Detect scheduler from environment:
- `bsub --help` exits 0 → LSF.
- `sbatch --help` exits 0 → Slurm.
- `nvidia-smi -L` returns ≥1 GPU → local.
- Else → emit error: "No compute backend detected; please run the templates manually."

## Output
Launcher invocation command with all parameters filled. Example for LSF:
```bash
bash templates/bsub.sh \
    --training-batch 0 \
    --seed 0 \
    --max-epochs 30000 \
    --signatures-csv ./signatures_output/ref_signatures/signatures.csv \
    --spatial-h5ad /path/to/spatial.h5ad \
    --output-name my_run
```

For chunked runs (`n_chunks > 1`), emit one invocation per `training_batch` index (0..n_chunks-1).

---

# Phase 10 — Aggregation (combine chunked outputs)

## Goal
If `n_chunks > 1`, combine per-chunk results into a single AnnData.

Skip if `n_chunks == 1`.

## Interactive
Confirm chunk outputs are present (`{output_dir}/{output_name}_chunk*/sp.h5ad`). Generate the aggregation notebook invocation.

## Autonomous
Same; if any chunk is missing → emit error with the missing chunk index.

## Aggregation logic (from [templates/step2_aggregate_chunks.ipynb](templates/step2_aggregate_chunks.ipynb))

```python
adata_full = sc.read_h5ad(spatial_h5ad_path)
for key in ['means', 'q05', 'q50', 'q95']:
    adata_full.obsm[key] = np.zeros((adata_full.n_obs, n_cell_types))
for i in range(n_chunks):
    adata_chunk = sc.read_h5ad(f"{output_dir}/{output_name}_chunk{i}/sp.h5ad")
    idx = adata_full.obs.index.get_indexer(adata_chunk.obs.index)
    for key in ['means', 'q05', 'q50', 'q95']:
        adata_full.obsm[key][idx] = adata_chunk.obsm[key]
    adata_full.uns[f'mod_batch{i}'] = adata_chunk.uns.get('mod', None)
adata_full.write(output_path)
```

Reference: [issue #356](https://github.com/BayraktarLab/cell2location/issues/356), [#375](https://github.com/BayraktarLab/cell2location/issues/375).

---

# Common errors — route to troubleshooting skill

The companion [cell2location-troubleshooting/SKILL.md](../cell2location-troubleshooting/SKILL.md) handles symptom-based debugging. Common patterns it covers:

- "ELBO is oscillating" → expected; keep training. [Issue #327](https://github.com/BayraktarLab/cell2location/issues/327).
- "OOM on `export_posterior`" → `use_quantiles=True`, reduce `sample_kwargs['batch_size']`. [Issue #278](https://github.com/BayraktarLab/cell2location/issues/278).
- "All cell types appear everywhere" → reference granularity too coarse. [Issue #395](https://github.com/BayraktarLab/cell2location/issues/395).
- "Comparing two samples" → train ONE joint model with `batch_key`. [Issues #389](https://github.com/BayraktarLab/cell2location/issues/389), [#396](https://github.com/BayraktarLab/cell2location/issues/396).
- "Model load fails / Pyro params not initialised" → `mod.train(max_epochs=1)` after `load()`. [Issues #365](https://github.com/BayraktarLab/cell2location/issues/365), [#404](https://github.com/BayraktarLab/cell2location/issues/404), [#421](https://github.com/BayraktarLab/cell2location/issues/421).

Load the troubleshooting skill alongside this one. When you can't resolve an issue, instruct the user to invoke `/cell2location-troubleshooting` to draft a clean issue.

---

# Anti-patterns the skill REFUSES (re-stated)

| Anti-pattern | Refusal | Citation |
|---|---|---|
| Mini-batch training for spatial mapping (`batch_size != None`) | REFUSE | [#356](https://github.com/BayraktarLab/cell2location/issues/356), supplement §1.3 |
| Log-transforming reference or spatial counts | REFUSE | [#386](https://github.com/BayraktarLab/cell2location/issues/386), supplement §2 |
| `num_samples=1000` on `n_obs > 100k` without `use_quantiles=True` | REFUSE | [#278](https://github.com/BayraktarLab/cell2location/issues/278), [#360](https://github.com/BayraktarLab/cell2location/issues/360) |
| Running [docs/notebooks/cell2location_tutorial.ipynb](../../../docs/notebooks/cell2location_tutorial.ipynb) on real-sized data | REFUSE; point to this skill | design rule |
| Training separate models per sample to compare conditions | REFUSE; use joint model with `batch_key` | [#389](https://github.com/BayraktarLab/cell2location/issues/389), [#396](https://github.com/BayraktarLab/cell2location/issues/396) |
| Computing per-cell-type per-gene per-location dense tensor on `n_obs > 100k` | REFUSE; compute per-chunk | [#375](https://github.com/BayraktarLab/cell2location/issues/375) |

---

<reference>

## Always-loaded reference

### Hyperparameter defaults (from supplement §1.2)

See [reference/fig_S27_hyperparameters.png](reference/fig_S27_hyperparameters.png) for the canonical decision tree. See [reference/fig_S1_workflow.png](reference/fig_S1_workflow.png) for the workflow overview.

| Hyperparameter | Default | Source |
|---|---|---|
| `N_cells_per_location` | Fig S27 (scalar from histology, or per-location N̂_s from segmentation, or cell-size fallback) | §1.2 item 1 |
| `detection_alpha` | 200 (low var) / 20 (high var) | §1.2 item 2 + Fig S27 |
| `n_groups` | 50 | §1.2 |
| `A_factors_per_location` | 7 | §1.2 |
| `B_groups_per_location` | 7 | §1.2 |
| `w_sf_mean_var_ratio` | 5 | §1.2 |
| `N_cells_mean_var_ratio` (v^n) | 1 (global) / 10 (per-location) | §1.2 |
| `N_cells_per_location_alpha_prior` | None (master) / 1000.0 (hires + segmentation) | hires-branch convention |
| `max_epochs` | 5k (small) / 30k (medium) / 30-80k (large per chunk) | §1.3 |
| `batch_size` | None (full-batch) | §1.3 — never override |
| `lr` | 0.002 | §1.3 |
| `use_quantiles` (export) | True for n_obs > 100k | issue #278 |

### Training (§1.3)

- ADVI in Pyro; ADAM `lr=0.002`.
- 20,000–50,000 iterations; stop at ELBO plateau.
- Late ELBO oscillations are normal — don't stop early.
- 1000 posterior samples; q05 is the paper's preferred point estimate.
- QC: `mod.plot_history(5000)` + posterior predictive log-log + spatial `y_s` consistency check.

## Suggested reading (for when you need more context)

These are NOT auto-loaded. Use the `Read` tool when the user asks "why" or you need a deeper answer.

- [reference/hyperparameters_extract.md](reference/hyperparameters_extract.md) — full paraphrase of supplement §1.2 + §1.3 + §1.4 + §2. *Read when:* you need the full default-by-default rationale, or when a user asks "why this default?".
- [reference/issue_corpus.md](reference/issue_corpus.md) — paraphrased vitkl-guidance harvest. *Read when:* user reports a symptom that doesn't match an anti-pattern above.
- [reference/c2l_supplement.pdf](reference/c2l_supplement.pdf) — full supplementary methods (17 pages text + figures). *Read when:* user asks about model features, new modalities, integration across experiments, multiple samples, mathematical guarantees — anything beyond hyperparameter defaults.
- [../../../cell2location/models/_cell2location_module.py:242](../../../cell2location/models/_cell2location_module.py#L242) — master-branch `forward()`. *Read when:* user asks why per-location `N_cells_per_location` doesn't seem to "stick" (answer: it's a soft prior in master).
- `cell2location/models/_cell2location_module.py:898` on `origin/hires_sliding_window` (fetch via `git show origin/hires_sliding_window:cell2location/models/_cell2location_module.py | sed -n '898,1100p'`) — hires-branch `forward()`. *Read when:* user is on hires with segmentation and wants to understand exactly how `n_s_cells_per_location` becomes a deterministic multiplier on `w_sf`.
- [../../../cell2location/models/_cell2location_WTA_module.py:247](../../../cell2location/models/_cell2location_WTA_module.py#L247) — WTA `forward()`. *Read when:* user is on Nanostring WTA/DSP and asks about negative-probe-binding or per-experiment gene calibration.
- [../../../cell2location/models/reference/_reference_module.py:178](../../../cell2location/models/reference/_reference_module.py#L178) — reference `RegressionModel` `forward()`. *Read when:* user asks why `batch_key` vs `categorical_covariate_keys` matter (answer: `batch_key` drives `detection_mean_y_e` + `s_g_gene_add`; `categorical_covariate_keys` drive per-gene tech regression `detection_tech_gene_tg`).

</reference>
