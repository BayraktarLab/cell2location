# cell2location recurring-issue corpus — paraphrased guidance from the maintainer

This file paraphrases the maintainer's (vitkl) recurring answers across the GitHub issue tracker. It is the canonical fallback the [spatial-mapping skill](../SKILL.md) and [cell2location-troubleshooting skill](../../cell2location-troubleshooting/SKILL.md) consult when a user reports a problem.

**Source**: systematic harvest across [BayraktarLab/cell2location](https://github.com/BayraktarLab/cell2location/issues) and [vitkl/cell2location_paper](https://github.com/vitkl/cell2location_paper). Snapshot date: 2026-04-26. New issues since then are NOT here; an agent matching a symptom should also run `gh search issues -R BayraktarLab/cell2location "<keywords>"` as a freshness check.

**Format**: each topic has: question, default/recommended, decision rule, why it matters, source issues.

---

## 1. Choosing `N_cells_per_location`

**Question**: How many cells in each spatial location?

**Default**: tissue-dependent; typically 5–30 cells per 10X Visium spot. For Visium-HD, similar but scale by bin size.

**How to choose**:
1. Count nuclei in histology (H&E or DAPI) across 10–20 regions at the same magnification as spots → average.
2. Provide as a scalar (per-tissue) OR per-location array (`shape: n_obs × 1`) from segmentation.
3. For early exploration: a rough estimate is fine; the model is robust to similar values.
4. Without histology: cell-size + capture-size formula (Visium 55 μm → 5; Slide-seq V2 → 1).

**Why it matters**: guides ABSOLUTE cell abundance estimation. Too low → underestimates; too high → overestimates. Essential for cross-section comparability.

**Sources**: [#344](https://github.com/BayraktarLab/cell2location/issues/344), [#383](https://github.com/BayraktarLab/cell2location/issues/383), [#394](https://github.com/BayraktarLab/cell2location/issues/394), supplement eq. 7 + Fig S27.

---

## 2. Setting `detection_alpha`

**Question**: How much does RNA detection sensitivity vary across locations within a batch?

**Default**:
- `detection_alpha = 200` — low within-slide variability (fresh-frozen single-sample Visium).
- `detection_alpha = 20` — high within-slide variability (FFPE, Cytassist, Visium-HD, older/degraded, multi-batch).

**How to choose**:
1. Inspect total-count distribution per location: histogram or violin plot per sample.
2. Uniform → `200`. Wide / many low-count regions → `20`.
3. Per-batch dict allowed: `{sample_a: 20, sample_b: 200}`.

**Why it matters**: controls per-location detection prior `y_s ~ Gamma(α, α/y_e)`. Wrong choice masks real technical effects (too high) or misattributes biology to noise (too low).

**Sources**: [#327](https://github.com/BayraktarLab/cell2location/issues/327), [#344](https://github.com/BayraktarLab/cell2location/issues/344), [#381](https://github.com/BayraktarLab/cell2location/issues/381), README + supplement §1.2.

---

## 3. Managing large spatial datasets (>40k spots)

**Question**: How to fit cell2location when data exceeds GPU memory?

**Default**: stratified chunks of ~40–72k spots; train independent models per chunk; merge in post-processing.

**How to choose**:
1. Estimate memory: `~18k genes × n_locations × 80GB A100 rule`.
2. If `n_obs > chunk_size_max`: split stratified-by-sample (random assignment across chunks; each chunk sees all samples).
3. Train each chunk independently (parallel jobs).
4. Merge: `adata_full.obsm[key][chunk_idx] = adata_chunk.obsm[key]`. Save chunk models as `adata.uns['mod_batch0']`, `mod_batch1`, etc.
5. **Critical**: each chunk uses `batch_size=None` (full-batch). **Never** mini-batch.

**Why it matters**:
- Prevents OOM crashes.
- Full-batch is dramatically more accurate AND faster in wall-clock than mini-batch (full-batch: 1–2h per chunk on A100; mini-batch: 10+ sec/iteration → days/weeks of training).

**Sources**: [#356](https://github.com/BayraktarLab/cell2location/issues/356), [#375](https://github.com/BayraktarLab/cell2location/issues/375), [#365](https://github.com/BayraktarLab/cell2location/issues/365), [#372](https://github.com/BayraktarLab/cell2location/issues/372), [#404](https://github.com/BayraktarLab/cell2location/issues/404), [#421](https://github.com/BayraktarLab/cell2location/issues/421).

---

## 4. Training duration / `max_epochs`

**Question**: How long to train?

**Default**:
- Moderate dataset (10k–40k locations): `max_epochs = 30000`.
- Very large (chunked, >100k): may need fewer per chunk (5k–10k).
- Very small (<5k): may need fewer (1k–5k).
- Convergence is data-dependent — inspect ELBO.

**How to choose**:
1. `mod.plot_history(1000)` (skip first 1k for visibility).
2. Stop when ELBO plateaus.
3. **Late-stage oscillations are normal and DO NOT harm results**. Higher stochasticity = a few parameters with intrinsic uncertainty.
4. Don't stop early on oscillations alone.
5. For smaller/cleaner data: 500–2000 may suffice.

**Why it matters**: oscillations reflect posterior uncertainty, not optimisation failure. Stopping early misses final parameter refinement.

**Sources**: [#327](https://github.com/BayraktarLab/cell2location/issues/327), [#375](https://github.com/BayraktarLab/cell2location/issues/375), supplement §1.3.

---

## 5. Reference signatures: source and format

**Question**: Where should reference signatures come from?

**Default ranking**:
- **Best**: scRNA-seq from same tissue (e.g. Allen Brain Atlas).
- **Good**: scRNA-seq from similar tissue/organism.
- **Poor**: MERFISH/imaging (lower genome coverage).
- **Avoid**: bulk RNA-seq (too averaged).

**Format**: pd.DataFrame, genes × cell_types, **linear count scale** (not log).

**Cell-count requirements**:
- Whole-tissue reference: ~40k cells.
- Per cell type: 100s of cells; minimum 40–50 for rare types.
- Genome-wide (not just marker genes).

**Critical**: do NOT pre-normalize. The `RegressionModel` handles batch/library-size correction. Pre-normalisation breaks the NB likelihood.

**Sources**: [#292](https://github.com/BayraktarLab/cell2location/issues/292), [#298](https://github.com/BayraktarLab/cell2location/issues/298), [#216](https://github.com/BayraktarLab/cell2location/issues/216), [#219](https://github.com/BayraktarLab/cell2location/issues/219), supplement §2.

---

## 6. Posterior sampling and abundance export

**Question**: How to summarize the posterior for downstream analysis?

**Default**: `use_quantiles=True` (NOW DEFAULT for large datasets).

**Critical**: **NEVER use `num_samples=1000` for n_obs > 100k**. Creates a dense matrix of `n_cell_types × n_obs × 1000` → OOM.

**How to invoke**:
```python
adata_vis = mod.export_posterior(
    adata_vis,
    use_quantiles=True,
    add_to_obsm=['means', 'q05', 'q50', 'q95'],
    sample_kwargs={
        'batch_size': mod.adata.n_obs // 4,
        'accelerator': 'gpu',
        'return_observed': False,
    },
    exclude_vars=['data_target'],
)
```

**Defaults**:
- Point estimate: `means` (most commonly used).
- Confidence: `q05` (lower), `q95` (upper). The paper used `q05` in all figures.
- Median: `q50` for non-symmetric posteriors.

**Sources**: [#278](https://github.com/BayraktarLab/cell2location/issues/278), [#281](https://github.com/BayraktarLab/cell2location/issues/281), [#360](https://github.com/BayraktarLab/cell2location/issues/360).

---

## 7. Comparing samples / batch effects

**Question**: How to compare cell abundance between two conditions?

**Default**: train ONE joint model with ALL samples concatenated; use `batch_key='sample'` in `setup_anndata`.

**How to choose**:
1. Concatenate into one AnnData; train one model.
2. **Never** train separate models per sample for comparison — each model estimates detection differently → abundance estimates not comparable.
3. For statistical testing: use quantiles from the single joint model.

**Why it matters**:
- Separate models break comparability.
- Joint model shares prior on batch effects → meaningful comparisons.
- Detection variation across batches is REAL and must be jointly accounted for.

**Sources**: [#389](https://github.com/BayraktarLab/cell2location/issues/389), [#396](https://github.com/BayraktarLab/cell2location/issues/396).

---

## 8. Input data normalization

**Question**: Should I normalize/log-transform before cell2location?

**Default**: **NO**. Raw counts only. Cell2location uses NB likelihood, which requires linear counts.

**How to choose**:
1. Start with raw filtered count matrices (e.g. from CellRanger / SpaceRanger).
2. Remove cells/spots with <100 UMIs, genes with <5 cells expressing them.
3. Keep raw counts.
4. For reference (RegressionModel): include `batch_key`; the model estimates batch effects automatically.
5. **Do NOT**: log-transform, CPM/TPM normalize, library-size scale.

**Sources**: [#386](https://github.com/BayraktarLab/cell2location/issues/386), supplement §2 + tutorial.

---

## 9. Handling very low count areas

**Question**: Exclude spots with very low RNA?

**Default**: yes — exclude `total_counts < 1000` (or dataset-median / 3).

**How to choose**:
1. `adata.obs['total_counts'] = adata.X.sum(1)`.
2. Histogram. If 50–70% of locations <1000 → consider replicating with higher-quality tissue.
3. If only outlier 5–10% locations low → filter out.

**Why it matters**: low RNA areas cannot distinguish cell types reliably. Including them adds noise without biological signal.

**Sources**: [#344](https://github.com/BayraktarLab/cell2location/issues/344), [#372](https://github.com/BayraktarLab/cell2location/issues/372).

---

## 10. Cell type granularity / reference annotation level

**Question**: How many cell types in the reference?

**Default**: 20–50 well-characterised cell types for comprehensive mapping.

**Anti-pattern — the "3-fingered glove on a 5-fingered hand"**:
- Too few types (e.g. 7) → all types appear everywhere; model has nowhere to put data not explained by the (too coarse) signatures.
- Rare types with <40–50 reference cells → noisy signatures → poor mapping.

**How to choose**:
1. Use biologically motivated subtypes.
2. Ensure 40–50 cells minimum per type.
3. If abundance maps don't match biology → may indicate insufficient granularity.

**Sources**: [#395](https://github.com/BayraktarLab/cell2location/issues/395).

---

## 11. Merging multiple trained models (chunked datasets)

**Question**: How to combine results from chunk-trained models?

**Default**: concatenate; copy `obsm` keys back to full adata at chunk indices.

**How to invoke**:
```python
# Create full-size adata with zeros for obsm
adata_full = sc.read_h5ad(original_path)
for key in ['means', 'q05', 'q50', 'q95']:
    adata_full.obsm[key] = np.zeros((adata_full.n_obs, n_cell_types))

# Copy chunk results back
for i, chunk_path in enumerate(chunk_paths):
    adata_chunk = sc.read_h5ad(chunk_path)
    idx = adata_full.obs.index.get_indexer(adata_chunk.obs.index)
    for key in ['means', 'q05', 'q50', 'q95']:
        adata_full.obsm[key][idx] = adata_chunk.obsm[key]
    # Save chunk model
    adata_full.uns[f'mod_batch{i}'] = adata_chunk.uns['mod']
```

**Don't**: try to merge `adata.uns['mod']` directly — model parameters are chunk-specific (location-dependent params like `w_sf`, `y_s`).

**Sources**: [#356](https://github.com/BayraktarLab/cell2location/issues/356), [#375](https://github.com/BayraktarLab/cell2location/issues/375).

---

## 12. RegressionModel — reference signature estimation

**Question**: How to estimate batch-corrected signatures from scRNA?

**Default**: `cell2location.models.RegressionModel`.

**How to invoke**:
```python
RegressionModel.setup_anndata(adata_ref, batch_key='sample', labels_key='cell_type')
# For multi-technology references, add:
RegressionModel.setup_anndata(adata_ref, batch_key='sample', labels_key='cell_type',
                                categorical_covariate_keys=['10x_kit', 'donor'])
mod_ref = RegressionModel(adata_ref)
mod_ref.train(max_epochs=300, accelerator='gpu')
inf_aver = mod_ref.export_posterior(adata_ref, select_quantiles=[0.5])
# Extract signatures (linear scale, batch-corrected)
signatures = mod_ref.adata.var[[c for c in mod_ref.adata.var.columns if 'q05_' in c]]
```

**Critical**: `batch_key` removes coarse batch effect; `categorical_covariate_keys` remove per-gene tech effects. Don't lump all tech covariates into `batch_key`.

**Sources**: [#216](https://github.com/BayraktarLab/cell2location/issues/216), [#219](https://github.com/BayraktarLab/cell2location/issues/219).

---

## 13. VisiumHD, Cytassist, and non-standard technologies

**Question**: Different settings for newer 10X Visium variants?

**Default**: `detection_alpha = 20` (less strict) for all of these.

**Why**: these technologies have higher technical variability than fresh-frozen Visium.

**Per-technology guidance**:
- **VisiumHD**: estimate `N_cells_per_location` from bin size (2/8/16 μm) and known cell density. Mandatory chunking (150k–650k bins per slide).
- **Cytassist**: high technical variability → `detection_alpha = 20`.
- **FFPE**: same as Cytassist.
- **Xenium, seqFISH+**: similar guidance; inspect total count distribution.

**Sources**: [#356](https://github.com/BayraktarLab/cell2location/issues/356), [#401](https://github.com/BayraktarLab/cell2location/issues/401), README.

---

## 14. Saving and loading models

**Question**: How to save / reload a trained model?

**Default**:
- Save: `mod.save('model_name', overwrite=True)`.
- Load: `mod = Cell2location.load('model_name', adata_vis)`.

**Critical caveat**: after `load()`, run `mod.train(max_epochs=1)` to instantiate Pyro parameters. **This is expected behaviour**, not a bug — Pyro params are lazily initialised on first training step. The single epoch is discarded.

**Save the adata separately**: `adata_vis.write('adata.h5ad')` — preserves abundance estimates in obsm.

**Sources**: [#365](https://github.com/BayraktarLab/cell2location/issues/365), [#402](https://github.com/BayraktarLab/cell2location/issues/402), [#404](https://github.com/BayraktarLab/cell2location/issues/404), [#421](https://github.com/BayraktarLab/cell2location/issues/421).

---

## 15. Expected gene expression per cell type (NCEM downstream)

**Question**: Can cell2location output per-cell-type expression per location for downstream analysis?

**Default**: yes, via `mod.compute_expected_nb_param_m_s_g()` or similar (version-dependent).

**For chunked models**: run per-chunk; concatenate as with abundance.

**Use case**: NCEM (neighborhood cell-type-conditioned expression), NMF colocation, custom downstream methods.

**Memory caveat**: per-cell-type per-gene per-location is a 3D tensor; large for big datasets. Compute per-chunk.

**Sources**: [#375](https://github.com/BayraktarLab/cell2location/issues/375), tutorial.

---

## 16. Amortised inference (future / experimental)

**Question**: Can I do fast inference on new data without retraining?

**Default — current state**: no. Full inference required per dataset; no out-of-sample prediction.

**Roadmap**: amortised (VAE-style) inference for 100k–1M+ locations in development. Experimental JAX backend.

**Sources**: README "Future development", experimental JAX branch.

---

## 17. ELBO is oscillating — should I stop?

**Quick answer**: **NO**. Late-stage ELBO oscillations are expected. Train to full `max_epochs`.

See topic #4 (Training duration).

---

## Anti-patterns the skill REFUSES (no exceptions)

These are baked-in refusals in the [SKILL.md](../SKILL.md):

1. Mini-batch training for spatial mapping (`batch_size != None`). Less accurate AND slower wall-clock than full-batch. [#356](https://github.com/BayraktarLab/cell2location/issues/356), supplement §1.3.
2. Log-transforming counts before cell2location. Breaks NB likelihood. [#386](https://github.com/BayraktarLab/cell2location/issues/386).
3. `num_samples=1000` on `n_obs > 100k` without `use_quantiles=True`. OOM. [#278](https://github.com/BayraktarLab/cell2location/issues/278).
4. Running the tutorial notebook on real-sized data.
5. Training separate models per sample to compare conditions. Use joint model with `batch_key`. [#389](https://github.com/BayraktarLab/cell2location/issues/389).
