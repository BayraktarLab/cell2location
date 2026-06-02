---
name: cell2location-context
description: "Capture and persist the scientific scope + technical decisions of a cell2location spatial-mapping project. Auto-discovers SPATIAL_MAPPING_CONTEXT.md in cwd / .claude/ / ~/.claude/plans/; if missing, runs a focused interview (--science = goals, reference, populations, success/failure criteria; --technical = Phases 1–8 decision sweep with gap-flagging). Called automatically by /spatial-mapping at the start (--science) and before launch (--technical). Standalone-invocable via /cell2location-context to update goals between runs."
user-invocable: true
---

# cell2location-context — persistent scope + decision memory for spatial mapping

This skill owns the **`SPATIAL_MAPPING_CONTEXT.md`** file: discovery, summarization, creation, updates. It is called automatically by [spatial-mapping/SKILL.md](../spatial-mapping/SKILL.md) (twice: once at the start with `--science`, once before launch with `--technical`) and can also be invoked standalone to edit goals without running the full mapping workflow.

The file is the durable project memory the workflow consults on every run: scientific goals, single-cell reference, target populations, success / failure criteria, plus the technical decisions made through Phases 1–8. It is also read by [cell2location-troubleshooting/SKILL.md](../cell2location-troubleshooting/SKILL.md) so symptom-matching can lean on the user-declared failure criteria.

---

## How this skill works

```
                       ┌─ file found → summarize → ask: Use / Re-interview / Skip
                       │
1. Auto-discover ─────┤
                       │
                       └─ no file  → ask: Interview / Import handoff / Skip
                                          │        │           │
                                          │        │           └→ emit “no context” warning marker; return "skipped"
                                          │        └→ ask path; persist stub with `## Source:` line; return its path
                                          └→ ask input-mode (one-time) → ask location → run rubric → persist → return path
```

The skill always returns one of: a file path the caller should load, or the string `"skipped"`.

## File location resolution (fixed candidate order)

Check, in order, on every invocation:

1. `$CWD/SPATIAL_MAPPING_CONTEXT.md` (top-level in the user’s spatial project — default for new files)
2. `$CWD/.claude/SPATIAL_MAPPING_CONTEXT.md` (hidden, project-scoped)
3. `~/.claude/plans/SPATIAL_MAPPING_CONTEXT_<dataset>.md` (centralized; `<dataset>` derived from the spatial AnnData filename, or asked from user)

First match wins. On **first creation**, ask the user via `AskUserQuestion` which of the three locations to save to. Default to option 1.

When called by `spatial-mapping`, the caller passes `$CWD` (the user’s data project) and, if known, the spatial AnnData filename (used for the `<dataset>` slug in option 3).

---

## Mode `--science` — scope interview

Run when invoked at the start of `spatial-mapping`, or standalone with no mode argument.

### Rubric (7 question groups)

Each group has a **primary question** + **grill-me-style follow-up triggers** for incomplete answers. Full text + recommended-answer examples in [reference/scope_interview_rubric.md](reference/scope_interview_rubric.md). Summary here:

| # | Group | Primary question | Follow-up trigger | Maps to |
|---|---|---|---|---|
| 1 | **Scientific goal** | Why are you running cell2location on this dataset? What biological question are you answering? | <20 words OR no concrete outcome named | Phase 0, all phases |
| 2 | **Single-cell reference** | What scRNA reference are you using? Path, source, tissue, cell count, label count. | Source unclear, tissue mismatch with spatial, <40 cells/type, <10 or >200 cell types | Phase 1 |
| 3 | **Annotation methodology** | How were the reference cell-type labels assigned? (markers / automated / manual / mixed) Confidence? Known weak spots? | "I just used what came with the atlas" — ask which subclusters were validated | Phase 1, troubleshooting |
| 4 | **Target populations** | Which specific cell-type populations MUST be spatially resolvable for this analysis to be useful? List as many as relevant. | Fewer than 3 populations named, or populations not in the reference labels | Phase 1 granularity check, Phase 3 |
| 5 | **Granularity check** | For each target population in #4, does the reference’s `labels_key` distinguish it from neighbours? (Agent auto-checks against `adata_ref.obs[labels_key].unique()` if reference is available; user confirms.) | Any target population lumped with others in the reference labels | Phase 1 (re-annotate?), Phase 3 |
| 6 | **Success criteria (3)** | What 3 concrete, measurable results would let you conclude "the analysis worked"? | Fewer than 3 listed; or any criterion that is vague ("looks reasonable") rather than measurable | Phase 8 QC, troubleshooting |
| 7 | **Failure criteria (5–10)** | What 5–10 outcomes must NOT happen for the result to be trustworthy? (Examples: "all cell types appear everywhere", "endothelial cells in white matter", "abundance varies 10× across visually similar regions".) | Fewer than 5 listed; or any criterion that does not describe an observable pattern in the output | Phase 8 QC, troubleshooting, refusal logic |

### Skill behavior

1. **Adapts to existing answers.** If the file is present (auto-discovery path) and the user chose "Re-interview", iterate group by group asking "Keep / Edit / Replace" for each.
2. **Grill follow-ups.** When the trigger fires, ask probing follow-ups one at a time via the chosen input mode — the same pattern as [/grill-me](/Users/kleshcv/.claude/claude-shared-skills/skills/grill-me/SKILL.md) but with the rubric kept inside this skill so questions stay cell2location-specific.
3. **Maps to workflow importance.** Each persisted answer is annotated with a `_→ affects Phase N_` tag from the table above. The user sees why the question was asked.
4. **Skill-driven recommendation.** For each question, the skill shows a recommended answer based on what is already known from the data (e.g. for #2 it can pre-fill the path if found in cwd; for #5 it can pre-fill the answer if the reference was inspected).

### Output

Writes the populated `## Scientific scope` block of the context file. Returns the file path.

---

## Mode `--technical` — implementation-completeness sweep

Run when invoked **after Phase 8 / before Phase 9 launch** of `spatial-mapping` (caller passes `--technical`). Sweeps the Phase-1-through-8 decisions, surfaces any unanswered slot, and cross-checks against the `## Scientific scope` block.

### What it does

1. **Loads** `## Scientific scope` (must exist; if not, route back to `--science` first).
2. **Reads** the in-progress notebook parameter dict the caller passes in (or, if missing, reads the workflow-state markdown the spatial-mapping skill emits between phases).
3. **Sweeps each technical slot** (one per row in the `## Technical decisions` block of the file schema). For each slot that is empty:
   - If the slot has a defensible default the skill applies it and notes the default in `## Outstanding gaps`.
   - If the slot has NO default (e.g. `labels_key` cannot be guessed), asks the user.
4. **Cross-checks scope vs decisions.** Examples:
   - User listed "distinguish oligodendrocyte subtypes" in #4 but `labels_key` resolved to a broad-cluster column → flag, link to [issue #395](https://github.com/BayraktarLab/cell2location/issues/395), ask to re-annotate.
   - Failure criterion "abundance varies 10× across visually similar regions" present but `detection_alpha=200` chosen → warn that `200` will not regularize away that variation, suggest `20`.
   - Per-location segmentation hinted at in #5 but Phase 6 chose master branch → flag, link to hires_sliding_window install instructions.
5. **Persists** `## Technical decisions` + `## Outstanding gaps` blocks.

Full cross-check matrix in [reference/technical_completeness_rubric.md](reference/technical_completeness_rubric.md).

### Output

Writes the populated `## Technical decisions` + `## Outstanding gaps` blocks. Returns the file path.

---

## Input-mode toggle (AskUserQuestion vs printed prompts)

On **first run**, ask once:

> How should I ask interview questions?
> 1. **Use the `AskUserQuestion` tool** — click-button choices, best in VSCode / Claude Desktop.
> 2. **Print questions in the conversation and pause for your typed response** — better for TUI / SSH / headless Claude Code where the AskUserQuestion UI is awkward.

Persist the choice as the first non-comment line under `## Input mode` at the top of `SPATIAL_MAPPING_CONTEXT.md`. Honor it on subsequent runs without re-asking. The user can edit the file or invoke `/cell2location-context --reset-input-mode` to change.

In **autonomous mode** (no `AskUserQuestion` tool exposed and no human attached), skip both interview modes; emit a notebook markdown cell `# No SPATIAL_MAPPING_CONTEXT.md found; ran with defaults. Provide a context file next time for guided runs.` and return `"skipped"`.

### Printed-prompt mode (mode 2)

Render each question as:

```
=== Scope question 3 / 7  — Annotation methodology  (→ affects Phase 1, troubleshooting) ===

How were the reference cell-type labels assigned? (markers / automated / manual / mixed)
What is your confidence in the labels? Are there known weak spots?

Recommended template:
  Method:      manual + marker-based, validated by ...
  Confidence:  high for major lineages, medium for subtypes
  Weak spots:  endothelial subtypes were not separated

>>> Type your answer below, then send the message <<<
```

Then stop and wait for the user’s next message. Treat the next user message as the answer; parse it; advance to the next question.

---

## Schema of `SPATIAL_MAPPING_CONTEXT.md`

The exact template lives in [templates/SPATIAL_MAPPING_CONTEXT.template.md](templates/SPATIAL_MAPPING_CONTEXT.template.md). Copy it into place on first creation and fill it incrementally as the user answers.

---

## Mapping from scope answers to workflow phases (quick reference)

| Scope answer | Spatial-mapping phase that consumes it |
|---|---|
| Goal (#1) | Phase 0 demo-vs-real branch; framing of all warnings |
| Reference path + source (#2) | Phase 1 `ref_h5ad_path` pre-fill; Phase 1 reference-too-small / too-coarse warnings |
| Annotation method + confidence (#3) | Phase 1 `labels_key` selection; Phase 1 categorical_covariate_keys suggestion |
| Target populations (#4) | Phase 1 granularity check; Phase 3 (whether per-location N̂ is needed for the populations of interest) |
| Granularity check (#5) | Phase 1 warning about re-annotation; Phase 6 (hires branch decision) |
| Success criteria (#6) | Phase 8 QC plot interpretation; troubleshooting routing |
| Failure criteria (#7) | Phase 4 `detection_alpha` choice (if abundance-variation criterion present); Phase 6 branch choice; troubleshooting routing |

The technical interview uses this table in reverse: for any technical decision whose corresponding scope answer is missing or stale, it re-asks.

---

## Companion skills

- [spatial-mapping/SKILL.md](../spatial-mapping/SKILL.md) — the caller. Invokes this skill at the start (`--science`) and before launch (`--technical`).
- [cell2location-troubleshooting/SKILL.md](../cell2location-troubleshooting/SKILL.md) — reads the `## Scientific scope` (esp. failure criteria) before suggesting symptom matches.

## Suggested reading (not auto-loaded)

- [reference/scope_interview_rubric.md](reference/scope_interview_rubric.md) — full question list, recommended-answer examples, grill-me follow-up triggers, examples of well-formed answers.
- [reference/technical_completeness_rubric.md](reference/technical_completeness_rubric.md) — the slot-by-slot technical sweep + the scope-vs-decision cross-check matrix.
- [templates/SPATIAL_MAPPING_CONTEXT.template.md](templates/SPATIAL_MAPPING_CONTEXT.template.md) — the empty file schema.
