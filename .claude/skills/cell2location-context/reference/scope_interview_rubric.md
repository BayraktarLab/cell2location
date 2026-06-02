# Scope interview rubric (`/cell2location-context --science`)

This is the full rubric the [SKILL.md](../SKILL.md) summarises in its 7-row table. Read this when running the interview.

For each group: **what to ask** → **recommended-answer template** → **grill-me follow-up triggers** → **how to persist** → **how it maps to the workflow**.

Two rendering modes: `AskUserQuestion` (button choices) and printed-and-pause (typed text). Both modes ask the same primary question; the printed mode also shows the "recommended template" verbatim so the user can edit-and-send. Honor the `## Input mode` line in `SPATIAL_MAPPING_CONTEXT.md`.

---

## 1. Scientific goal — _affects Phase 0, all phases_

### Primary question
> Why are you running cell2location on this dataset? What biological question are you answering? 2–4 sentences. Concrete is better than abstract.

### Recommended answer template
```
We want to map <cell types of interest> across <tissue> sections from <experimental
groups>. The biological question is whether <population A> is enriched in <region X>
vs <region Y>, and whether that enrichment changes between <condition 1> and
<condition 2>. The outcome will feed a <downstream figure / hypothesis / drug
target validation step>.
```

### Grill-me follow-up triggers
- **<20 words.** Ask: "What's the biological question you'd ask a reviewer the result *answers*?"
- **No populations named.** Ask: "Which cell-type populations would you need to see distributed differently for the result to be interesting?"
- **No downstream named.** Ask: "What figure or decision in your paper / project does the cell2location output feed into?"
- **Mentions a clinical decision but no biological mechanism.** Ask: "What spatial pattern would distinguish a true-positive prediction from a false-positive in this clinical context?"

### Persist into
`## Scientific scope → ### Scientific goal` (free text).

### Maps to
- Phase 0: helps decide demo-vs-real branch and framing of all warnings.
- Phase 1 granularity warnings: anchored to "populations of interest" mentioned here.

---

## 2. Single-cell reference — _affects Phase 1_

### Primary question
> What scRNA-seq reference are you using? Provide path, source, tissue/organism, number of cells, number of cell-type labels.

### Recommended answer template
```
- Path:        /path/to/snrna_reference.h5ad   (or URL)
- Source:      <atlas name / in-house / published — DOI>
- Tissue:      <e.g. mouse cortex + hippocampus, 8 wk C57BL/6>
- N cells:     <e.g. 41,000>
- N labels:    <e.g. 47> in column `<labels_key>`
```

### Grill-me follow-up triggers
- **Tissue does not match the spatial tissue.** Ask: "Your reference is <X> but your spatial is <Y>. Is there a tissue-matched reference available? If not, name the cell types you expect to be present in <Y> that aren't in <X>."
- **Source unclear.** Ask: "Was this reference produced in-house or downloaded? If downloaded, what's the DOI or atlas name?"
- **<40 cells per type in any target population.** Warn: "Cell types with <40 cells will have noisy signatures and unreliable spatial estimates." Ask if they have a denser version.
- **<10 cell-type labels.** Warn: "cell2location works best with 20–50 well-characterised types. With <10, all types tend to appear everywhere (issue #395)."
- **>200 cell-type labels.** Warn: "Many types with overlapping signatures cause identifiability issues. Consider merging clusters."

### Persist into
`## Scientific scope → ### Single-cell reference` (structured fields).

### Maps to
- Phase 1: pre-fills `ref_h5ad_path`; drives reference-too-small / too-coarse warnings.

---

## 3. Annotation methodology — _affects Phase 1, troubleshooting_

### Primary question
> How were the reference cell-type labels assigned? (markers / automated / manual / mixed) What is your confidence in the labels? Are there known weak spots?

### Recommended answer template
```
Method:      manual annotation by <person/lab>, validated by <markers / overlap with
             prior atlas / orthogonal modality>
Confidence:  high for major lineages (neuronal / glial); medium for subtypes;
             low for <specific lineage>
Weak spots:  endothelial subtypes were not separated; clustered as a single label.
```

### Grill-me follow-up triggers
- **"I just used what came with the atlas."** Ask: "Which subclusters in this atlas were validated by orthogonal evidence vs assigned by clustering only?"
- **No weak spots mentioned.** Probe: "Are there cell types in your target list (§4) for which the reference label is the result of a single marker gene? Those tend to be the noisy ones in spatial maps."
- **Mixed methods but no per-lineage confidence.** Ask: "For each target population, is your confidence high / medium / low? Anything <high should be called out in the failure criteria."

### Persist into
`## Scientific scope → ### Annotation methodology`.

### Maps to
- Phase 1: drives `labels_key` selection and the `categorical_covariate_keys` recommendation (low-confidence subtypes argue for keeping the level coarser).
- Troubleshooting: when a population doesn't appear where expected, this section is the first place to check.

---

## 4. Target populations — _affects Phase 1 granularity check, Phase 3_

### Primary question
> Which specific cell-type populations MUST be spatially resolvable for this analysis to be useful? List as many as relevant — at least 3.

### Recommended answer template
```
- <population A>: why it matters — <e.g. defines the cortical layer boundary>.
- <population B>: why it matters — <e.g. marks the disease vs control divergence>.
- <population C>: why it matters — <e.g. negative control: should be absent from white matter>.
- <population D>: …
```

### Grill-me follow-up triggers
- **Fewer than 3 populations named.** Ask: "If only one cell type was correctly mapped and the others were noise, would the analysis still be useful? If not, which 3+ populations do you need correctly mapped?"
- **Populations not in the reference labels.** Cross-check against `adata_ref.obs[labels_key].unique()` if available. If a population is missing, ask: "<population> is not in the reference's `<labels_key>`. Is it lumped with another label, or absent entirely?"
- **Only positive populations mentioned.** Probe: "What's a *negative* control population — a cell type whose presence in a specific region would be a sign of a wrong map?"

### Persist into
`## Scientific scope → ### Target populations` (bulleted list).

### Maps to
- Phase 1: drives the granularity warnings.
- Phase 3: if any target population needs sub-spot resolution, that argues for per-location N̂ + hires branch.

---

## 5. Granularity check — _affects Phase 1 re-annotation decision, Phase 6_

### Primary question
> For each target population in §4, does the reference's `labels_key` distinguish it from neighbours?

### Skill-driven assistance
If the reference is available (path from §2), inspect:

```python
import scanpy as sc
adata_ref = sc.read_h5ad(ref_h5ad_path)
labels = adata_ref.obs[labels_key].unique().tolist()
for pop in target_populations:
    matches = [l for l in labels if pop.lower() in l.lower()]
    print(pop, "→", matches or "MISSING")
```

Pre-fill the table the user confirms.

### Grill-me follow-up triggers
- **Target population lumped with others.** Ask: "<A> is lumped with <X> in `<labels_key>`. Options: (a) accept the lumped label and lose subtype resolution, (b) re-annotate at a finer level using a marker-based or supervised tool (Celltypist / scANVI). Which?"
- **Target population missing entirely.** Ask: "<A> is not in the reference. Is there a different column in `adata.obs` that has it, or do you need a different reference?"

### Persist into
`## Scientific scope → ### Granularity check` (table).

### Maps to
- Phase 1: re-annotation warning if needed.
- Phase 6: if sub-spot resolution is needed (e.g. distinguishing two subtypes that often co-occupy a spot), the hires branch becomes attractive.

---

## 6. Success criteria — 3 measurable outcomes — _affects Phase 8 QC, troubleshooting_

### Primary question
> What 3 concrete, measurable results would let you conclude "the analysis worked"? Each criterion should be observable in the output (a number, a spatial pattern, a statistical test).

### Recommended answer template
```
1. I can rank cortical layers L1–L6 by relative GABAergic interneuron subtype
   abundance, with ≥3 subtypes showing layer-specific enrichment at p<0.01.
2. The known marker-gene-based spatial pattern of <population A> is recovered:
   q05 abundance map of <A> visually matches the marker gene `<marker>` ISH
   reference in <atlas link>.
3. Inter-sample variability in <population B> abundance correlates with the
   experimental condition (rho > 0.5 across N=<N> samples).
```

### Grill-me follow-up triggers
- **Fewer than 3.** Ask one by one until 3 are listed.
- **Vague criterion** ("looks reasonable" / "matches biology"). Ask: "How would you measure that? What number / pattern would let you decide yes vs no?"
- **Criterion that depends on cell2location's own output being self-consistent** ("ELBO converges"). Reject: "Convergence is necessary but not sufficient. Restate as a biological pattern that should be visible."

### Persist into
`## Scientific scope → ### Success criteria` (numbered 1–3).

### Maps to
- Phase 8 QC: the success criteria become explicit checks in the QC notebook.
- Troubleshooting: failure to meet a criterion routes to a specific symptom in the corpus.

---

## 7. Failure criteria — 5–10 outcomes that must NOT happen — _affects Phase 4, Phase 6, refusal logic, troubleshooting_

### Primary question
> What 5–10 outcomes must NOT happen for the result to be trustworthy? Each criterion describes an observable wrong-pattern in the output.

### Recommended answer template
```
1. All cell types appear everywhere — "3-fingered glove on a 5-fingered hand"
   (issue #395). Diagnostic: max(q05 abundance per type per location) < 2× median.
2. Endothelial cells appear in white matter where there is no vasculature.
3. Total cell abundance varies >10× across visually similar regions of the same
   sample (sign that detection_alpha is mis-set).
4. Negative cell types — types not present in the reference but spatially detected.
5. Posterior predictive QC log-log plot deviates from y=x by more than one decade.
6. Per-sample mean abundance of <population A> varies more across replicates than
   the biological signal of interest (rho < condition-rho).
7. <…>
```

### Grill-me follow-up triggers
- **Fewer than 5.** Probe one by one — show the user the corpus examples above as templates.
- **Vague criterion** ("looks wrong"). Ask: "Describe the spatial pattern that would make you mistrust the result. What would a reviewer flag?"
- **No "abundance varies across visually similar regions" criterion.** Ask: "If two replicates of the same tissue type showed 10× different total abundance, would you trust the map? If not, add that as a criterion." (This drives Phase 4 detection_alpha.)
- **No "cell types appear everywhere" criterion.** Add it as a default — this is the most common failure mode (issue #395).

### Persist into
`## Scientific scope → ### Failure criteria` (numbered 1–N).

### Maps to
- Phase 4 `detection_alpha`: if "abundance varies 10×" is a criterion, `detection_alpha=20` is mandatory.
- Phase 6 branch: if "subtype mixing" is a criterion AND segmentation is available, hires branch is mandatory.
- Refusal logic: the skill should refuse to launch if any chosen technical decision would obviously violate a stated failure criterion.
- Troubleshooting: when the user later reports a symptom, match it against this list first.

---

## How the skill picks the next question

After each answer is captured, before moving to the next group:

1. **Update the in-memory context.** Reflect what's now known.
2. **Re-evaluate the trigger list.** Some triggers reference later groups — e.g. "populations not in the reference labels" is a trigger for group 4 but requires knowing the reference from group 2.
3. **Adapt subsequent recommended templates.** If group 2 revealed a 41k-cell mouse-brain reference with 47 labels, group 4's template should use the actual label names as examples, not generic placeholders.
4. **Track group completion.** A group is "done" when (a) the primary question is answered AND (b) no triggers are firing.

When all 7 groups are done, write the file and return.

---

## Examples of well-formed scope blocks

See [examples/](.) — not yet populated; add real examples here as the skill is used in practice.
