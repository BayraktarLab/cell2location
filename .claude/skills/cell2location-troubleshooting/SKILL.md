---
name: cell2location-troubleshooting
description: "Diagnose cell2location problems (convergence failures, OOM, anomalous abundance, biological interpretation). Routes to existing GitHub issues when they match; drafts a clean `gh issue create` body with diagnostic metadata when not. Invoked LAST after main spatial-mapping skill attempts have failed OR when user is dumping an error. Companion to ../spatial-mapping/SKILL.md."
user-invocable: true
---

# cell2location-troubleshooting — diagnose problems and file clean issues

This skill helps when something goes wrong with cell2location, or when a user has a question heavy on biological interpretation. It is the companion skill to [../spatial-mapping/SKILL.md](../spatial-mapping/SKILL.md).

**Convention**: when the main `spatial-mapping` skill loads, this skill should be loaded too. But this skill is also usable **standalone** — invoke `/cell2location-troubleshooting` from any project.

## What this skill does

1. **Phase 1 — Match the user's symptom** against the harvested issue corpus at [../spatial-mapping/reference/issue_corpus.md](../spatial-mapping/reference/issue_corpus.md). The corpus paraphrases ~25 recurring vitkl answers across the GitHub tracker. Most user-reported problems already have a documented answer.

2. **Phase 2 — Search GitHub** when the symptom doesn't match the local corpus. Uses `gh search issues -R BayraktarLab/cell2location` to find newer issues the corpus may not have.

3. **Phase 3 — Draft a `gh issue create` body** when neither match. Pre-fills the diagnostic metadata vitkl normally asks for: environment, data shape, hyperparameters used, ELBO trajectory, full error trace. Reduces back-and-forth.

The skill **NEVER auto-submits** issues. It drafts; the user reviews and submits.

## When to invoke

- (a) The main `spatial-mapping` skill instructed you to (after exhausting its anti-pattern checks).
- (b) The user is dumping an error or unexpected result.
- (c) The user explicitly says "help me file a cell2location issue".
- (d) The question is heavily about biological interpretation — route to [discourse.scverse.org](https://discourse.scverse.org/c/ecosytem/cell2location/42).

## Phase 0 — Intent classification

Classify the user's question:

- **Convergence / training problem** (ELBO not decreasing, NaN, OOM during training) → Phase 1.
- **Posterior / export OOM** → Phase 1.
- **Anomalous abundance** (all cell types everywhere, missing cell type, bad maps) → Phase 1. If no corpus match, route to discourse.
- **Biological interpretation** ("why does my biology look like X?") → route to discourse directly; do NOT file a GitHub issue.
- **Novel modality / integration / multi-sample method question** → Phase 1, then Phase 2.
- **"Help me file an issue"** → Phase 3 directly.

## Phase 1 — Corpus match

Read the issue corpus: [../spatial-mapping/reference/issue_corpus.md](../spatial-mapping/reference/issue_corpus.md). The corpus is organised by topic:

1. Choosing `N_cells_per_location`
2. Setting `detection_alpha`
3. Managing large spatial datasets (>40k spots)
4. Training duration / `max_epochs`
5. Reference signatures: source and format
6. Posterior sampling / abundance export OOM
7. Comparing samples / batch effects
8. Input data normalization
9. Handling very low count areas
10. Cell type granularity / reference annotation level
11. Merging multiple trained models (chunked)
12. RegressionModel — reference signature estimation
13. VisiumHD, Cytassist, non-standard technologies
14. Saving and loading models
15. Expected gene expression per cell type
16. Amortised inference (future / experimental)
17. ELBO oscillation (should I stop training?)

For each topic, the corpus has: question, default/recommended, decision rule, why it matters, source issue links.

When you match a symptom to a topic:
- Quote the relevant guidance to the user.
- Link the source issue numbers so the user can `gh issue view` for full context.
- DO NOT continue to Phase 2 or 3 — the answer is already public.

## Phase 2 — GitHub fallback

If the symptom doesn't match the local corpus (e.g., a new issue type that postdates the 2026-04-26 corpus snapshot):

```bash
gh search issues "<symptom keywords>" \
    -R BayraktarLab/cell2location \
    --state all \
    --limit 5 \
    --json number,title,body,state,author
```

For high-signal answers, filter by `--author vitkl`.

Present the top 3 matches to the user as a numbered list with titles + URLs. Ask them to confirm whether their issue matches any of these. If yes → done. If no → Phase 3.

## Phase 3 — Issue draft

Compose a `gh issue create` body with the diagnostic metadata vitkl normally has to request. Use this template:

```markdown
## Environment
- cell2location version: <run `pip show cell2location` and paste version>
- Branch installed from: master / hires_sliding_window / other (specify)
- scvi-tools version: <run `pip show scvi-tools` and paste version>
- Python version: <python --version>
- PyTorch version: <pip show torch>
- Pyro version: <pip show pyro-ppl>
- GPU model + memory: <e.g. NVIDIA A100 80GB>

## Data
- Spatial technology: 10X Visium / Visium-HD / Cytassist / Slide-seq V2 / Stereo-seq / Nanostring WTA/DSP
- Number of locations (n_obs): ...
- Number of genes (n_vars): ...
- Number of batches (samples): ...
- Reference signature source: scRNA-seq atlas / cluster_averages / user-provided
- Number of cell types (factors): ...
- Total counts distribution per sample:
  - 10th percentile: ...
  - 50th percentile (median): ...
  - 90th percentile: ...
  - 90/10 ratio: ...   (informs detection_alpha choice)

## Hyperparameters used
- `N_cells_per_location`: <scalar value, OR column name + summary stats if per-location array>
- `detection_alpha`: <value>
- `n_groups` (R): ...
- `A_factors_per_location`: ...
- `B_groups_per_location`: ...
- `use_proportion_factorisation_prior_on_w_sf`: <True / False>
- `use_n_s_cells_per_location_limit`: <True / False>
- `use_per_cell_type_normalisation`: <True / False>
- `max_epochs`: ...
- `batch_size`: None (full-batch) ← MUST be None

## Symptom
<concise description; paste error trace VERBATIM if applicable>

## ELBO history
<paste output of `mod.plot_history(5000)` — screenshot or text summary of first / last 1000 epochs>
- Looks plateaued? <yes / no>
- Late oscillations? <yes / no — see issue #327: oscillations are expected>

## What I have already tried
<list anti-patterns from the spatial-mapping skill the user has avoided; reference any relevant SKILL.md phases>

## Reproducer
<minimal-as-possible code snippet that reproduces the issue, OR pointer to the executed papermill notebook>
```

Then instruct the user to run:

```bash
# Create a draft issue (review BEFORE submitting):
gh issue create \
    -R BayraktarLab/cell2location \
    --title "<short title>" \
    --body-file <(cat <<'BODY'
<paste the filled-in template above>
BODY
)
```

### Privacy guard

- The diagnostic template asks for SHAPES and SUMMARY STATS, not raw data.
- Do NOT auto-include `adata.obs.head()`, `adata.X[:10]`, or any user data in the issue body.
- The user adds biology / dataset names manually if they want; the skill never adds them automatically.

## Routing to discourse instead of GitHub

For these question categories, point the user at [discourse.scverse.org](https://discourse.scverse.org/c/ecosytem/cell2location/42) instead of GitHub:

- "Why does cell type X appear in region Y in my data?" — biological interpretation.
- "How do I interpret the co-location groups?" — biological interpretation.
- "Which cell types should be in my reference?" — biology / experimental design.

GitHub issues are for code / model bugs. Biology questions are better-suited to community discussion.

## Anti-patterns this skill REFUSES

- Auto-submitting issues without user review.
- Including raw user data (`adata.obs.head()`, `adata.X.toarray()`, etc.) in issue bodies.
- Filing duplicate issues when the corpus or `gh search` already has a match.
- Filing GitHub issues for pure biology questions (route to discourse).

---

<reference>

## Issue corpus (shared with main skill)

The corpus is at [../spatial-mapping/reference/issue_corpus.md](../spatial-mapping/reference/issue_corpus.md). Snapshot date: 2026-04-26. New issues since then are NOT in the corpus; use Phase 2 (`gh search`) to find them.

## Suggested reading

These are NOT auto-loaded. Use the `Read` tool when needed:

- [../spatial-mapping/reference/issue_corpus.md](../spatial-mapping/reference/issue_corpus.md) — full corpus, by topic.
- [../spatial-mapping/reference/hyperparameters_extract.md](../spatial-mapping/reference/hyperparameters_extract.md) — supplement §1.2-§1.4 + §2 paraphrase. *Read when:* the user's problem points at a hyperparameter choice and you need the canonical default rationale.
- [../spatial-mapping/SKILL.md](../spatial-mapping/SKILL.md) — main skill's anti-patterns block. *Read when:* you suspect the user is hitting a known anti-pattern.

</reference>
