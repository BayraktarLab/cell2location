# Technical-completeness rubric (`/cell2location-context --technical`)

Sweeps the Phase-1-through-8 technical decisions, fills any gaps with defaults, asks for the un-default-able ones, and **cross-checks the chosen decisions against the `## Scientific scope` block** (especially failure criteria). Persists into `## Technical decisions` and `## Outstanding gaps`.

Read [scope_interview_rubric.md](scope_interview_rubric.md) first to understand what the scope block contains.

---

## Sweep procedure

For each technical slot in the table below:

1. **Load current value** from (a) the in-progress notebook parameters the caller passed in, or (b) the existing `## Technical decisions` block if re-running, or (c) "EMPTY" if neither.
2. **If EMPTY and a default exists**: apply the default, record `default applied` in `## Outstanding gaps`.
3. **If EMPTY and no default**: ask the user. Honor the input-mode preference.
4. **If set**: confirm with the user (one-tap "keep" in AskUserQuestion mode, or "Press enter to keep" in printed mode).
5. **Run the cross-check** in the next section. If a check fires, route to the relevant remediation.

---

## Slot table

| Phase | Slot | Default if EMPTY | Source rule |
|---|---|---|---|
| 1 | `labels_key` | none — must ask | Phase 1 rubric in spatial-mapping/SKILL.md |
| 1 | `batch_key` (ref) | `sample` if present | Phase 1 |
| 1 | `categorical_covariate_keys` | `[]` | Phase 1; ask if multi-tech detected |
| 1 | `continuous_covariate_keys` | `[]` | Phase 1 |
| 1 | gene filters | `cell_count_cutoff=15, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12` | Fig S1 |
| 1 | `max_epochs` (ref) | `min(round(20000/n_cells * 400), 400)` | Phase 1 |
| 2 | `spatial_h5ad_path` | none — must ask | Phase 2 |
| 2 | `spatial_technology` | autodetect from `adata.uns['spatial']` etc. | Phase 2 autonomous rules |
| 2 | `batch_key` (spatial) | `sample` if present | Phase 2 |
| 2 | `total_counts_min / max` | 1000 / 200000 | Phase 2 |
| 2 | gene-filter thresholds (spatial) | `cell_count_cutoff=15, cell_percentage_cutoff2=0.15, nonz_mean_cutoff=1.11` | Phase 2 |
| 3 | `N_cells_per_location` value | Fig S27 cell-size fallback by technology | Phase 3 autonomous tree |
| 3 | `N_cells_per_location_alpha_prior` | `1` (master) / `1000` (hires) | Phase 6 |
| 3 | `N_cells_mean_var_ratio` | `1` (global N̂) / `10` (per-location N̂_s) | §1.2 |
| 4 | `detection_alpha` | autodetect: `20` if any sample 90/10 ratio > 10× OR FFPE/Cytassist/Visium-HD; else `200` | Phase 4 |
| 5 | `n_chunks` | from `chunk_size_max = available_gpu_memory_bytes / (n_vars × 8 × 4)` | Phase 5 |
| 5 | stratification | stratified random by `batch_key` | Phase 5 |
| 6 | branch | `hires_sliding_window` if per-location N̂_s chosen; else `master`; `WTA` if Nanostring | Phase 6 |
| 6 | `use_proportion_factorisation_prior_on_w_sf` | `True` on hires, else absent | Phase 6 |
| 6 | `use_n_s_cells_per_location_limit` | `True` on hires, else absent | Phase 6 |
| 7 | `n_groups` (R) | 50 | §1.2 |
| 7 | `A_factors_per_location` | 7 | §1.2 |
| 7 | `B_groups_per_location` | 7 | §1.2 |
| 7 | `w_sf_mean_var_ratio` | 5 | §1.2 |
| 7 | `use_per_cell_type_normalisation` | `False` | §1.2 |
| 8 | `max_epochs` (spatial) | small (<5k loc) → 5000; medium (10k–40k) → 30000; large → 30k–80k per chunk | §1.3 |
| 8 | `lr` | 0.002 | §1.3 — never override |
| 8 | `batch_size` | `None` (full-batch) | §1.3 — REFUSE if set |
| 8 | `use_quantiles` (export) | `True` if `n_obs > 100000` else `True` | issue #278 |
| 8 | `add_to_obsm` | `['means', 'q05', 'q50', 'q95']` | §1.3 |
| 9 | backend | autodetect (`bsub` → LSF, `sbatch` → Slurm, `nvidia-smi -L` → local, else error) | Phase 9 |

---

## Cross-check matrix (scope ↔ technical)

For each scope answer, the matching technical-decision check:

| Scope element | Technical check | Action if violated |
|---|---|---|
| Target population lumped in `labels_key` (granularity check §5) | Phase 1 `labels_key` resolves to coarse column | Flag in `## Outstanding gaps`: "Target population <X> is lumped with <Y> in chosen `labels_key=<L>`. See issue #395. Recommend: re-annotate to a finer level OR remove <X> from target list." |
| Failure criterion "all cell types appear everywhere" | Reference has <10 unique labels in `labels_key`, OR labels_key chosen at very coarse level | Flag: "Failure criterion #N risks triggering. Reference has only <K> labels. Re-annotate at finer level (Celltypist / scANVI) before training." |
| Failure criterion "abundance varies 10× across visually similar regions" | `detection_alpha=200` chosen | Flag + suggest: "Failure criterion #N would not be regularised by `detection_alpha=200`. Set `detection_alpha=20` (per Fig S27 lower flow) to allow the model to absorb the within-sample variation." |
| Failure criterion "subtype mixing within spots" + segmentation hinted at (§3 mentions n_cell/occupancy) | Phase 6 chose `master` branch | Flag: "Per-location segmentation is available but master branch was selected. Per-location N̂_s is only a soft prior in master. Install hires_sliding_window to make it a deterministic multiplier. See SKILL.md Phase 6." |
| Success criterion "abundance correlates with experimental condition" | `n_chunks > 1` AND chunk assignment is NOT stratified by `batch_key` | Flag: "Cross-sample comparisons require stratified chunking. Confirm `n_chunks > 1` chunks are stratified so each chunk contains all samples." |
| Success criterion "marker-gene spatial pattern recovered" | None directly; record as a Phase 8 QC todo | Add to `## Outstanding gaps`: "QC step: overlay q05 abundance of <population> with marker `<gene>` ISH reference; visually confirm match." |
| Failure criterion "posterior predictive QC log-log deviates from y=x by >1 decade" | Phase 8 `mod.plot_QC()` not in notebook | Flag: "Failure criterion #N requires a QC plot that isn't in the notebook. Add `mod.plot_QC(summary_name='q05')` after training." |
| Failure criterion "negative cell type appears" | None directly | Add to `## Outstanding gaps`: "QC step: enumerate the top-3 cell types in regions where they should be absent; cross-check against tissue annotation." |
| Reference has multi-tech (§3 method = "mixed: 10X v2 + v3") | Phase 1 `categorical_covariate_keys` does NOT include the technology column | Flag: "Reference is multi-tech but per-gene tech regression is off. Add the technology column to `categorical_covariate_keys` to enable per-gene detection adjustment (see `_reference_module.py:178`)." |
| Annotation confidence is "low" on a target population (§3) | None directly | Add to `## Outstanding gaps`: "Annotation confidence is LOW on <population>. Treat the spatial map of this population as preliminary; consider orthogonal validation." |

---

## Output

Three things are written, all into the same `SPATIAL_MAPPING_CONTEXT.md`:

1. `## Technical decisions` — every slot filled with either the user's value, the default, or `<unanswered>` (rare; only when the user explicitly skipped a slot that has no default).
2. `## Outstanding gaps` — every cross-check that fired, every default that was applied, every QC step the success/failure criteria implicitly require.
3. A return value to the caller (`spatial-mapping`): the file path, AND a boolean `safe_to_launch` (`False` if any cross-check that has a hard remediation — re-annotation, branch switch, detection_alpha mismatch — fired without being addressed).

If `safe_to_launch=False`, the caller (spatial-mapping Phase 8.5 → Phase 9) must ask the user one more time before launching: "Cross-check flagged N issues. Launch anyway, fix and re-run, or abort?"
