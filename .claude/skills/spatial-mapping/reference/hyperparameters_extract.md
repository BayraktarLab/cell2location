# cell2location hyperparameters — paraphrase of supplementary methods §1.2–§1.4 + §2

This file is the structured paraphrase of the cell2location 2022 paper's supplementary methods. It is the authoritative reference for hyperparameter choice in the [SKILL.md](../SKILL.md) operating manual. Full supplement is at [c2l_supplement.pdf](c2l_supplement.pdf).

For human readers: every default below is from Kleshchevnikov 2022 supplement; the skill applies them verbatim unless a user override is recorded.

---

## §1.2 — Three data-dependent hyperparameters

There are exactly **three** hyperparameters that depend on the user's data. The remaining priors keep their defaults (table below).

### 1. `N̂` — Expected cell abundance per location (`N_cells_per_location`)

Tissue-level global estimate informing the prior on cell-count per spot. **The single most important user-provided value**. See [Fig S27](fig_S27_hyperparameters.png) for the decision tree.

Decision tree (verbatim from Fig S27):

```
Paired histology / DAPI image available?
├── YES → Can you segment nuclei per location (e.g. CNN / DAPI segmentation)?
│        ├── YES (advanced) → Provide per-location N̂_s, with N̂_s + 0.1 pseudocount;
│        │                    v^n = 10 (high confidence in segmentation).
│        └── NO  → Estimate per-tissue average by manually counting nuclei in
│                  10–20 locations (10X Loupe browser). Single tissue-level N̂.
└── NO  → Same-tissue histology (not paired) available?
         ├── YES → Same manual-count procedure on similar tissue.
         └── NO  → Use cell-size + capture-size formula:
                   • 10X Visium (55 μm capture)    → N̂ ≈ 5
                   • Slide-Seq V2 (10 μm bead)     → N̂ ≈ 1
                   • Visium-HD (2/8/16 μm bin)     → scale by area
                   • Cytassist Visium              → ≈ 5 (low confidence)
                   • Nanostring WTA/DSP            → per-region segmentation
```

For all paper analyses, a single tissue-level estimate was used. The orange-highlighted path in Fig S27 (manual counting → single scalar) is applicable to most 10X Visium users.

### 2. `α^y` — Regularisation of within-experiment RNA detection variation (`detection_alpha`)

Controls the prior on per-location detection efficiency variability:

```
y_s ~ Gamma(α^y, α^y / y_e)
```

where `y_e` is the per-batch mean sensitivity (latent).

Decision rule (Fig S27, lower flow):

```
Strong within-batch variation in total RNA count NOT explained by tissue containing more cells?
├── YES → α^y = 20  (less strict regularisation; allows y_s to vary widely per spot)
│         Common in:
│           - FFPE Visium
│           - Cytassist
│           - Visium-HD
│           - Older / degraded human samples
│         Note: tissues with intrinsic regions of high cell density
│         (hippocampus, gut lymphoid follicle) explain higher RNA by cell density,
│         not by detection variability — α^y = 20 still appropriate.
└── NO  → α^y = 200 (strict; y_s tightly clustered around y_e)
          Common in:
            - Fresh-frozen single-sample Visium
            - High-quality single-batch experiments
```

If unsure: inspect the per-location `total_counts` distribution; if its 90/10 percentile ratio > 10× → use `α^y = 20`.

### 3. `μ^y` — Expected detection sensitivity (auto-derived)

NOT a user-set value. Computed automatically from data by the model wrapper:

```
μ^y = (Σ_s Σ_g d_{s,g} / S) / (N̂ × Σ_f Σ_g g_{f,g} / F)
```

Where `S` = total locations, `F` = total cell types, `d_{s,g}` = spatial counts, `g_{f,g}` = reference signatures. The skill does not surface this to the user.

---

## Other priors — keep at defaults (per §1.2)

| Hyperparameter | Symbol | Default | What it controls |
|---|---|---|---|
| Cell types per location | `Â` = `A_factors_per_location` | **7** | Average # of cell types contributing per spot |
| Tissue zones per location | `B̂` = `B_groups_per_location` | **7** | Average # of co-located cell-type compartments per spot |
| Co-located cell type groups (R) | `n_groups` | **50** | Latent dimensionality of factorisation; model finds 3–7 substantive groups |
| Co-abundance prior strength | `v^w` = `w_sf_mean_var_ratio` | **5** | Medium; increasing → over-smoothing; decreasing → cell types independent |
| N̂ prior strength | `v^n` = `N_cells_mean_var_ratio` | **1** (global N̂) / **10** (per-location N̂_s) | How tightly N̂ constrains the model |

Notes:
- The model is **robust** to a range of `Â` and `B̂` (sensitivity analysis Fig S5). Default 7 is in the middle of the safe range.
- The embryo workflow used `B̂ = 5`. This is a project-specific deviation; do not adopt it as a default for other tissues without a stated reason.
- For laser-capture microscopy (LCM) or other methods with substantially varying location size, provide per-location `N̂_s` with `v^n = 10`.

---

## §1.3 — Inference

- **Framework**: Pyro / ADVI (Automatic Differentiation Variational Inference). Mean-field Gaussian over softplus-transformed parameters.
- **Optimiser**: ADAM, `lr = 0.002`.
- **Iterations**: 20,000–50,000 typically. **Stop at ELBO plateau** (`mod.plot_history(5000)` — skip first 5k for visibility).
- **ELBO oscillations late in training are expected and OK** — do not stop on first oscillation. Some parameters have intrinsic uncertainty; oscillations reflect that, not optimization failure. [GitHub issue #327](https://github.com/BayraktarLab/cell2location/issues/327).
- **Posterior samples**: 1,000 from the variational distribution → mean, std, q05, q95.
- The paper used `q05` for all figures (slightly more accurate than mean for absolute cell abundance).

### Recommended QC after training

1. **Posterior predictive check**: `log10(μ_{s,g} + 1)` vs `log10(d_{s,g} + 1)` — fit should be near-diagonal across the count range.
2. **`y_s` consistency**: spatial distribution of estimated `y_s` should resemble total RNA count `Σ_g d_{s,g}` across locations; AND total cell abundance `Σ_f w_{s,f}` should be consistent with histology (areas with more cells → higher abundance).

### Prior predictive check (optional)

For new tissues/technologies, optionally sample from the prior (without observed data) to verify the priors give plausible synthetic counts before fitting. Implemented in pymc3 backend.

---

## §1.4 — Nuclei segmentation (optional, advanced)

When per-location `N̂_s` is desired:

1. H&E or DAPI image of the spatial tissue section.
2. CNN ensemble (32 nets, Unet/FPN, trained on dsb2018) — [yozhikoff/segmentation](https://github.com/yozhikoff/segmentation).
3. Predicted nuclei masks → kd-tree-assigned to spatial-array locations.
4. Per-location nuclei count column in `adata.obs`.

**Effectiveness caveat** (not in the paper, but documented in the skill): per-location N̂_s is only **deterministically effective** in the `hires_sliding_window` branch with both `use_proportion_factorisation_prior_on_w_sf=True` AND `use_n_s_cells_per_location_limit=True`. In master, N̂_s acts as a soft prior that the model can drift from. The embryo workflow's `n_cell_occupancy = occupancy × N_nuclei × scaling` formula corrects for partly-empty spots and is required for accurate per-location abundance. See [../SKILL.md](../SKILL.md) Phase 6 for branch selection.

---

## §2 — Reference signature estimation

Two approved methods + a user-provided fallback:

### Method 1 (recommended): NB regression via `RegressionModel`

`cell2location.models.RegressionModel(adata, ...)` — full Bayesian regression accounting for batch + per-gene technology effects. Robust across multi-batch, multi-technology references.

User inputs:
- `labels_key` — cell-type column (drives `per_cluster_mu_fg`, the output signatures).
- `batch_key` — batch column (drives per-batch detection `detection_mean_y_e` + additive background `s_g_gene_add`).
- `categorical_covariate_keys` — list of additional categorical covariates (e.g. `donor`, `10x_kit`); drives per-gene per-tech regression `detection_tech_gene_tg`. **Use this for multi-technology references** (e.g. 10X v2 + v3 + Smart-seq); do NOT lump them all into `batch_key`.

### Method 2 (Smart-seq 2 fallback): hard-coded cluster averages

`cell2location.cluster_averages.get_cluster_averages(adata, labels_key)` — fast, no batch correction. Use when:
- Reference is from one batch / one technology (minimal technical heterogeneity).
- Smart-seq 2 — distributional assumptions of Method 1 (NB) don't fit Smart-seq 2 well.

### Method 3: user-provided signatures

Any `pd.DataFrame` with shape (genes × cell_types) in **linear count scale** (not log). The model's NB likelihood expects linear counts.

### Gene-selection rule (Fig S1 panel)

Two-cutoff selection at the start of reference estimation:
1. Selecting genes detected at count > 0 in many cells (> 5% of cells), AND
2. Selecting genes detected at count > 0 in a few cells (5% > cells > 10) but with mean expression across non-zero cells slightly > 1.0 (e.g. > 1.12).

Defaults: `cell_count_cutoff=15`, `cell_percentage_cutoff2=0.03`, `nonz_mean_cutoff=1.12`. Adjust for very small or very sparse references.

---

## Joint multi-sample modelling — benefits

Three reasons cell2location is run on all samples jointly rather than per-sample:

1. **Detection-sensitivity normalisation across batches** — `y_e` hierarchical prior on `y_s` allows direct comparison of cell abundance across tissue sections with varying sequencing depth.
2. **Sharing gene-technology effect `m_g`** — improves ability to distinguish low sensitivity from zero abundance.
3. **Sharing co-location factorisation `x_{r,f}`** — increases sensitivity to which cell types co-locate across experiments.

**Never** train separate models per sample to compare conditions. Use one joint model with `batch_key='sample'`.
