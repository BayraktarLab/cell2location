# Spatial mapping project context
_Created: <YYYY-MM-DD> by /cell2location-context_
_Last updated: <YYYY-MM-DD>_

> This file is owned and maintained by the `cell2location-context` skill. The `spatial-mapping` skill auto-discovers it on every run (see candidate paths in [.claude/skills/cell2location-context/SKILL.md](.claude/skills/cell2location-context/SKILL.md)). Edit by hand or via `/cell2location-context`.

## Input mode

<!-- One of: prefer-asktool | prefer-printed-prompts.  Controls how the skill asks questions on subsequent runs. -->
prefer-asktool

---

## Scientific scope
_Filled by `/cell2location-context --science`. Re-run that command to update._

### Scientific goal
<!-- → affects Phase 0 framing and all phase warnings -->
<2–4 sentences. What biological question does this analysis answer? Why are you spatially mapping cell types into this tissue?>

### Single-cell reference
<!-- → affects Phase 1 (`ref_h5ad_path`, reference-too-small / too-coarse warnings) -->
- **Path:** <absolute path OR URL>
- **Source:** <atlas name | in-house | published — cite>
- **Tissue / organism:** <e.g. "mouse brain (cortex + hippocampus), 8 wk C57BL/6">
- **Number of cells:** <N>
- **Number of cell-type labels:** <K> (column `<labels_key>` in `adata.obs`)

### Annotation methodology
<!-- → affects Phase 1 (`labels_key` selection, `categorical_covariate_keys` suggestion) and troubleshooting -->
- **Method:** <markers | automated (scANVI / Celltypist) | manual | mixed>
- **Confidence:** <high | medium | low>, possibly per-lineage
- **Known weak spots:** <e.g. "endothelial subtypes were not separated">

### Target populations
<!-- → affects Phase 1 granularity check, Phase 3 (per-location N̂?) -->
Populations that MUST be spatially resolvable for this analysis to be useful:

- **<population A>** — why it matters: <sentence>
- **<population B>** — why it matters: <sentence>
- **<population C>** — why it matters: <sentence>
- …

### Granularity check
<!-- → affects Phase 1 (re-annotate?), Phase 6 (hires branch decision) -->
For each target population above, is it distinguished from neighbours in the reference's `labels_key`?

| Target population | In `labels_key`? | Notes |
|---|---|---|
| <A> | YES / NO / LUMPED-WITH-<X> | <e.g. "Lumped with B in lvl-3; need lvl-5 re-annotation"> |
| <B> | YES | |
| … | | |

### Success criteria (3 measurable outcomes)
<!-- → affects Phase 8 QC interpretation, troubleshooting routing -->
The analysis is "done" if all three of these are true:

1. <e.g. "I can rank cortical layers L1–L6 by relative GABAergic interneuron subtype abundance, with at least 3 subtypes showing layer-specific enrichment p<0.01">
2. <…>
3. <…>

### Failure criteria (5–10 outcomes that must NOT happen)
<!-- → affects Phase 4 (detection_alpha), Phase 6 (branch choice), refusal logic, troubleshooting -->
The result is untrustworthy if any of these is observed:

1. <e.g. "All cell types appear everywhere — \"3-fingered glove on a 5-fingered hand\" pattern (issue #395)">
2. <e.g. "Endothelial cells appear in white matter where there is no vasculature">
3. <e.g. "Total cell abundance varies >10× across visually similar regions of the same sample">
4. <e.g. "Negative cell type — type not present in the reference — appears in tissue">
5. <e.g. "Posterior predictive QC log-log plot deviates from y=x by more than one decade">
6. <…>
7. <…>

---

## Technical decisions
_Filled by `/cell2location-context --technical`. Re-run that command after each `/spatial-mapping` session to refresh._

### Phase 1 — Reference signatures
- **labels_key:** <…>
- **batch_key:** <…>
- **categorical_covariate_keys:** <…>
- **continuous_covariate_keys:** <…>
- **gene filters:** defaults | overridden as <…>
- **max_epochs:** <…>

### Phase 2 — Spatial QC
- **spatial_h5ad_path:** <…>
- **technology:** <visium | visium-hd | cytassist | slide-seq-v2 | stereo-seq | nanostring-wta>
- **batch_key:** <…>
- **total_counts_min / max:** <…> / <…>
- **gene-filter thresholds:** defaults | overridden as <…>

### Phase 3 — N_cells_per_location
- **S27 path:** Q1=<yes/no> → Q2=<yes/no> → Q3=<yes/no>
- **N̂ value:** scalar <X> | per-location column `<n_cell_occupancy>`
- **N_cells_per_location_alpha_prior:** 1 | 1000 (hires)
- **N_cells_mean_var_ratio (v^n):** 1 | 10

### Phase 4 — detection_alpha
- **value:** 20 | 200 | per-batch dict <{…}>
- **rationale:** <e.g. "90/10 percentile ratio = 14× → high variability → 20">

### Phase 5 — Chunking
- **n_chunks:** <…>
- **chunk_size:** <…>
- **stratification:** stratified by `<batch_key>` (each chunk sees all samples)

### Phase 6 — Branch
- **branch:** master | hires_sliding_window | WTA
- **use_proportion_factorisation_prior_on_w_sf:** True | False (hires only)
- **use_n_s_cells_per_location_limit:** True | False (hires only)

### Phase 7 — Other hyperparameters
- **overrides from defaults:** none | <list>

### Phase 8 — Training + posterior
- **max_epochs:** <…>
- **lr:** 0.002
- **batch_size:** None (full-batch)
- **use_quantiles:** True | False
- **posterior add_to_obsm:** ['means', 'q05', 'q50', 'q95']

### Phase 9 — Compute
- **backend:** LSF (bsub) | Slurm (sbatch) | local | Jupyter interactive

---

## Outstanding gaps
_Auto-flagged by `/cell2location-context --technical`._

- <e.g. "No paired histology — using Fig S27 fallback N̂=5; flagged as low-confidence. Mitigation: redo with segmentation if absolute abundance estimates are needed for the publication.">
- <e.g. "Target population 'oligodendrocyte subtypes' (see Scientific scope §Target populations) is LUMPED-WITH-OPC in current `labels_key`; recommend re-annotation before treating subtype maps as biology.">
- <…>
