# Flow Smart Segments — Implementation Report

**Date:** 2026-09-30
**Scope:** Smart Segments V1, Milestones 0–8 (ML/data/model/export). Android integration (Milestones 9–13) not started.
**Repositories:** `seshuthota/Flow` (app) and `seshuthota/Flow-SponsorML` (ML pipeline)
**ML repo head at time of writing:** `02784dea` (`origin/main`, pushed)

---

## 1. Executive summary

The Smart Segments data, supervision, dataset, gate, model, export and evaluation
infrastructure is implemented, tested (185 Python tests green) and committed. Both
required audits exist; the frozen dataset is built with zero leakage; a multi-head
model was trained on GPU and exported to a hash-pinned ONNX bundle.

**The honest headline: the trained model is not usable, and the cause is the
supervision data, not the code.** Held-out evaluation shows the model marks
roughly 99% of every video as every category (span precision ≈ 0). Training had no
negative examples to learn from — the supervision contract deliberately excludes
`UNKNOWN` tokens from the loss, so each category's only supervised tokens were its
own positives, and the model converged to firing everywhere. This is exactly the
failure mode the plan anticipates in §21–§23, and it means every release gate in
§41/§42 remains unmet.

No release claim is made anywhere in this work. Details and the required fix are in
§10 and §11.

---

## 2. Repositories and environment

| Item | Value |
| --- | --- |
| ML repo | `/home/curious/Documents/hermes-projects/YT_experiments/Flow-SponsorML` |
| App repo | `/home/curious/Documents/hermes-projects/YT_experiments/Flow` |
| Python (ML tooling) | `.venv` (data extras; no torch) |
| Python (training/export) | conda env `llm`: Python 3.12.12, torch 2.10.0+cu128, transformers 4.57.3, datasets 4.8.4, accelerate 1.12.0, onnx 1.22.0, onnxruntime 1.30.0 |
| GPU | NVIDIA RTX 5060 Laptop, 8.1 GB, CUDA available |
| Dependency note | `llm` had onnxruntime already; only `onnx 1.22.0` was added, with no upgrades to torch/transformers/numpy/datasets/accelerate |
| Data sources | `sb-mirror/sponsorTimes.csv` (21,660,746 rows), ScriptSmith subtitles (62,818 videos), Xenova snapshot (98,219 videos) |

The ML pipeline was moved out of the Flow repo by commit `3cdda0f0`; `ml/` and
`sb-mirror/` are now gitignored in Flow. All new ML work lives in Flow-SponsorML.

---

## 3. Milestone status

| Milestone | State | Evidence |
| --- | --- | --- |
| 0 — upstream sync | Done on a dedicated branch, validated. Not merged to `main`. | `sync/upstream-20260929`; tag `pre-upstream-merge-20260929` |
| 1A — raw data audit | Done | `reports/smart_segments_raw_data_audit.json` |
| 1B — trainability audit | Done | `reports/smart_segments_trainability_audit.json` |
| 2 — canonical annotations | Done | `reports/smart_segment_annotations_manifest.json` |
| 3 — builders consume canonical | Done | `cba6295d` |
| 4 — supervision contract | Done | `ba39fdde` |
| 5 — frozen dataset | Done | `reports/smart_segments_windows_manifest.json` |
| 5b — manual benchmark | Queue built and reserved; **human review pending** | `reports/smart_segments_benchmark_review_manifest.json` |
| 6 — release gates | Contract frozen | `reports/smart_segments_gate_contract.json` |
| 7 — multi-head model | Trained + exported + evaluated | `reports/smart_segments_ettin_17m_metrics.json`, `reports/smart_segments_export_report.json`, `reports/smart_segments_video_evaluation.json` |
| 8 — bundle manifest | ML side done and validated; Android loader pending | `artifacts/smart_segments/manifest.json` |
| 9–13 | Not started (shadow mode, action resolver, actions, V2/V3) | — |

---

## 4. Milestone 0 — upstream sync (Flow repo)

- Upstream remote configured: `https://github.com/A-EDev/Flow.git`.
- Compared `main` (`0064e7e9`) with `upstream/main` (`33a5b5cd`): merge-base
  `9dc49e6d`, upstream **1 commit ahead** (`#1170`, localized Takeout history /
  import-safe recommendations / YouTube likes import).
- Safety tag `pre-upstream-merge-20260929` created at `0064e7e9`.
- Merged cleanly (no conflicts) on branch `sync/upstream-20260929` (`ce9c2e2c`).
- Validation on the merge: `ktlintCheck`, `:app:testGithubDebugUnitTest`,
  `:app:testFossDebugUnitTest`, `:app:assembleGithubDebug`, `:app:assembleFossDebug`,
  `:app:compileGithubDebugAndroidTestKotlin` → **BUILD SUCCESSFUL**. No device pass.
- `main` was not modified and nothing was pushed from this work. Since then the
  project owner added `docs(agents): add the upstream sync playbook` to `main`
  (`525c46d7`); `upstream/main` is not yet merged into `main`.

---

## 5. Milestone 1A — raw data availability audit

Command: `sponsor-detection smart-segments audit-raw`
Output: `reports/smart_segments_raw_data_audit.json` (81 s)

161,002 transcript-bearing videos (ScriptSmith 62,818; Xenova 98,219; overlap 35),
60,694 of the ScriptSmith videos English, 13,915 channels with eligible segments,
and 68,672 videos carrying more than one category.

| Category | Videos | Segments | Positive hours | Channels |
| --- | ---: | ---: | ---: | ---: |
| sponsor | 116,407 | 197,405 | 3,524.63 | 13,576 |
| selfpromo | 36,992 | 52,989 | 528.13 | 3,742 |
| interaction | 37,518 | 46,600 | 183.17 | 2,832 |
| outro | 32,746 | 34,319 | 219.05 | 2,930 |
| intro | 30,800 | 37,308 | 187.60 | 2,667 |
| preview | 9,197 | 10,358 | 95.85 | 1,826 |
| filler | 4,841 | 11,346 | 98.79 | 1,324 |
| music_offtopic | 406 | 722 | 9.73 | 158 |
| hook | 168 | 183 | 1.74 | 82 |

Cross-category overlap is substantial and must survive normalization:
`selfpromo+sponsor` 9,614 pairs, `intro+sponsor` 9,267, `interaction+selfpromo` 3,698,
`outro+sponsor` 3,256, `interaction+sponsor` 2,339, `interaction+outro` 2,323.

Mirror-wide context: sponsor 5,222,102 rows, intro 3,720,624, filler 2,992,290,
outro 2,879,758, selfpromo 2,150,895, interaction 1,707,623, preview 729,943,
music_offtopic 552,951, hook 84,740. Of 21,660,746 rows, 391,230 were eligible
audited rows joined to a transcript.

**Go/No-Go:** Go for V1 sponsor/selfpromo/interaction. `hook` is not trainable
(168 videos, 1.7 h). `music_offtopic` is thin. Caption provenance (manual vs auto)
is not recorded in the ScriptSmith snapshot and is reported as unavailable rather
than guessed.

---

## 6. Milestone 1B — trainability audit

Command: `sponsor-detection smart-segments audit-trainability` (2,117 s)
Output: `reports/smart_segments_trainability_audit.json`

385,770 windows, 368,127,765 tokens, 59,609 videos. The only alignment rejection
reason is `no_cue_overlap` (the segment falls in a caption gap).

| Category | Eligible | Aligned | Unaligned | Pos. windows | Pos. tokens | Token fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sponsor | 86,755 | 84,355 | 2,400 | 81,271 | 17,368,344 | 4.72% |
| selfpromo | 15,653 | 15,220 | 433 | 13,931 | 1,861,795 | 0.51% |
| interaction | 8,591 | 8,351 | 240 | 8,542 | 600,350 | 0.16% |
| preview | 5,380 | 5,170 | 210 | 4,810 | 642,999 | 0.17% |
| intro | 10,803 | 8,478 | 2,325 | 7,765 | 320,373 | 0.09% |
| outro | 9,004 | 7,824 | 1,180 | 8,044 | 410,959 | 0.11% |
| filler | 4,725 | 4,526 | 199 | 3,663 | 582,892 | 0.16% |
| music_offtopic | 355 | 231 | 124 | 153 | 29,823 | 0.008% |
| hook | 104 | 103 | 1 | 107 | 19,404 | 0.005% |

V1 categories align at ~97%. `intro`/`outro` have higher unaligned counts because
they are frequently music/animation with no speech. **Confirmed negatives are 0
for every category** — no exhaustive-coverage evidence source exists in the corpus.
Both reports pin the mirror, subtitles, metadata and Xenova hashes, the eligibility
and normalization/window/alignment versions, and the tokenizer revision plus
`tokenizer.json` SHA-256 (`9fd55248…`).

A real defect was found and fixed during this milestone: token counting used
window-local character spans against transcript-global token offsets, which zeroed
every non-initial window. Fixed with a regression test that fails without the fix.

---

## 7. Milestones 2–4 — canonical layer and supervision

**Milestone 2 — canonical annotations** (`e37769b8`)
`smart-segment-annotations/1`. One row per audited mirror annotation with category,
timestamps, votes, lock, quality metadata, action type, service, eligibility verdict,
rejection flags and source provenance; ineligible rows retained. Output
`data/smart_segments/smart_segment_annotations.parquet`:
20,040,926 rows, 15,937,454 eligible, 1,182,691,098 bytes,
SHA-256 `c6b4e23e71b1f8686b97caacb9cc1f1be65068e3c186b76f12c4a9acdcdf238e`.
The sponsor subset reproduces the legacy sponsor-only count exactly (4,532,860
eligible). Cross-category overlaps are preserved; within-category dedup is an
opt-in versioned policy that currently only exposes `preserve`. The existing
sponsor-only builder and outputs are untouched.

**Milestone 3 — builders consume canonical** (`cba6295d`)
The ScriptSmith and Xenova builders read the canonical table via
`load_canonical_annotations`. Spans are category-tagged and the dedup is per
category, so cross-category overlaps survive. Regression tests prove canonical
(not mirror) sourcing, per-span category identity, preserved overlap, and the
sponsor-only view of the shared field.

**Milestone 4 — supervision contract** (`ba39fdde`)
`smart-segment-supervision/1`. Per-category `POSITIVE` / `NEGATIVE_CONFIRMED` /
`UNKNOWN`, per-token loss masks that zero `UNKNOWN`, and provenance-mandatory
confirmed negatives. A positive for one category never implies a negative for
another. This is the correct design for a positive-unlabeled multi-label problem —
and §10 shows what happens when there are no confirmed negatives to complement it.

---

## 8. Milestone 5 — frozen dataset and benchmark

**Frozen dataset** — `reports/smart_segments_windows_manifest.json`

| Split | Rows | Videos | Channels | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 212,682 | 48,326 | 11,242 | `6e343bd627d0…` |
| validation | 23,123 | 5,389 | 1,386 | `1a80675eb566…` |
| test | 24,852 | 5,754 | 1,359 | `cb31940b5ebc…` |

261,657 windows total; 94,164 positive and 167,069 ordinary-negative; 7,375
training windows positive for more than one category. **Zero** video, channel and
exact-campaign leakage across every split pair. Each row carries explicit
`category_supervision` entries `{category, state, evidence_id}` as well as
`category_spans`.

**Benchmark** (`ca52dcd5`, `00100eb6`) — `reports/smart_segments_benchmark_review_manifest.json`

A hard ordering constraint surfaced: the benchmark must reserve channels *before*
training excludes them, or the two draw from the same finite channel pool and no
disjoint benchmark can exist (the first attempt selected 0 videos). The sampler now
writes a reserved exclusion directory (`data/smart_segments/benchmark_reserved`,
SHA-256 `2c835a5fcc10…`) and training excludes it.

59,398 candidates; **140 channel-disjoint videos selected**: sponsor 40,
selfpromo 30, interaction 30, content-negative 40, each with its full reconstructed
transcript. The dataset rebuild excludes exactly those 140 videos/channels
(0 overlap). The review queue is `annotation_pending` — **human review has not been
done**, so `freeze-benchmark` (which would derive confirmed negatives) has not run.

---

## 9. Milestones 6–8 — gates, model, export

**Milestone 6 — release gates** (`826dad85`) — `reports/smart_segments_gate_contract.json`
A frozen `SponsorBaselineContract` derived from the released sponsor manifest
(model SHA-256 `81184081034207ff…`, span F1@0.5 **0.7683**), plus non-inferiority
margins, per-category minimum reviewed evidence, an automatic-action contract
(sponsor enabled, selfpromo/interaction disabled) and a performance matrix.
Validation fails loudly if a blocking metric has no margin. Per-hour metrics are
left null until the benchmark records whole-video durations, rather than guessed.
`confidence_interval_method = wilson_95`.

**Milestone 7 — multi-head model** (`c6029618`, `37df7b9c`, `63875c4d`, `78966ba0`, `6394cc23`)
A shared Ettin 17M encoder with one independent BILOU head per category, projected to
`[batch, sequence, category, bilou]`. The sponsor head initialises from the released
combined checkpoint; new heads are random. Loss is cross entropy over exactly the
tokens and categories the contract marked known. Encoder and head use separate
learning rates (1e-5 / 2e-4).

Training (GPU, `llm` env): 2 epochs, 13,294 steps, bf16. Checkpoint
SHA-256 `5c5e673fad20bcd0…`, train loss 0.0931, validation loss 0.0331, test loss
0.0343. The first process was killed externally at step 6,000 (no exception, empty
log) and resumed from `checkpoint-6000`.

**Milestone 8 — bundle manifest** (`74949f25`, `62905f4f`)
`smart_segment_manifest` validates the pinned manifest hash first, then schema and
runtime versions, a contiguous category axis, the ordered `O/B/I/L/U` label axis and
the `[batch, sequence, category, bilou]` output layout, and verifies every artifact
hash. Export (`export-smart-segments`) produces a self-contained bundle via the
TorchScript exporter (`dynamo=False`, so no `onnxscript` dependency):

- `smart_segments.fp32.onnx` — 67,391,428 B, SHA-256 `f9e09fd725a4…`
- `smart_segments.int8.onnx` — 28,784,208 B (Gather-quantized), SHA-256 `1ae4bacd87ce…`
- tokenizer, decoder config, `manifest.json` (all hashes verified)

**Parity** (measured against a real held-out window): fp32 vs PyTorch max
|Δlogit| 1e-5, **argmax agreement 1.0**; int8 vs PyTorch max |Δlogit| 0.126,
**argmax agreement 1.0**.

---

## 10. Held-out evaluation — the decisive result

Command: `sponsor-detection smart-segments evaluate-videos` (`02784dea`)
Output: `reports/smart_segments_video_evaluation.json`

Full-transcript inference (windows → ONNX `segment_logits` → per-category BILOU
decode → character spans → timestamps → stitched intervals) over the 140
channel-disjoint benchmark videos (62.78 review-hours), CPU int8, 72.94 s
(1.92 videos/s).

| Category | Presence P / R | Span P / R @0.5 | Predicted vs expected | FP s/hour |
| --- | --- | --- | ---: | ---: |
| sponsor | 0.286 / 1.000 | 0.0135 / 0.0185 | 224,342 s vs 5,787 s | 3,481 |
| selfpromo | 0.279 / 1.000 | 0.0000 / 0.0000 | 224,225 s vs 2,245 s | 3,537 |
| interaction | 0.271 / 1.000 | 0.0049 / 0.0204 | 223,890 s vs 705 s | 3,556 |

Every category is predicted across ~99% of all reviewed time. Missed seconds are
near zero; false positives are enormous.

**Cause.** With `UNKNOWN` tokens excluded from the loss, each category's only
supervised tokens are its own positive span tokens. Nothing penalises firing, so
training converges to firing everywhere. This is a supervision-data problem. It is
not a decoding, threshold or quantization artifact: fp32 and int8 agree exactly
(argmax agreement 1.0), and thresholds cannot help a model that is confident and
wrong.

Reference labels here are SponsorBlock weak labels, so this run cannot establish
release-grade precision anyway; it is a genuine held-out measurement against weak
labels and is sufficient to show the model is not shippable.

---

## 11. Findings, limitations and required next steps

**Findings**
1. The audits confirm V1 categories are plentiful and alignable (~97%), but contain
   **no confirmed negatives** — the single most important gap.
2. The multi-head architecture, canonical layer, masked-loss targets, export and
   parity all work end to end; the failure is data, not plumbing.
3. The benchmark must reserve channels before the training build; this is now
   enforced in configuration.
4. A metric bug reported false 1.0 token F1 early; it was replaced with
   positive-token recall plus counts, and evaluation moved to span metrics.

**Not proven / limitations**
- No release gate is met. Sponsor non-inferiority, per-category precision and
  automatic-action eligibility are all unproven and must not be claimed.
- Thresholds are uncalibrated defaults; the decoder config is explicitly
  `"calibrated": false`.
- The benchmark has not been human-reviewed, so confirmed negatives do not exist.
- Android integration (loader migration, shadow mode, action resolver, actions,
  V2/V3) is untouched.
- No device/performance-gate measurement was taken (the eval timings are CPU
  desktop numbers, not the device matrix).

**Recommended next steps, in order**
1. Decide the negative-supervision policy. The defensible minimal option is a
   **sponsor-only weak-negative policy** (treat a video with no sponsor annotation
   as an O-negative for the sponsor head) — this is how the released sponsor model
   was trained, so it is fair for that one category and would let the sponsor head
   reach usable precision. It does **not** fix selfpromo/interaction, which need a
   genuinely larger reviewed negative set. This decision belongs to the owner,
   because the plan forbids treating missing annotations as negatives without a
   stated evidence contract.
2. Review the 140-video benchmark queue and run `freeze-benchmark` to produce the
   evaluation set and its confirmed negatives.
3. Re-train with negatives present, then calibrate per-category display/action
   thresholds against the frozen benchmark and evaluate against
   `smart_segments_gate_contract.json`.
4. Only after the gates pass, start Android Milestones 9–10 (shadow mode without
   playback changes, then the action resolver).

---

## 12. How to reproduce

```bash
# ML tooling (data audits, datasets, gates, benchmark)
cd Flow-SponsorML
.venv/bin/sponsor-detection smart-segments audit-raw
.venv/bin/sponsor-detection smart-segments audit-trainability
.venv/bin/sponsor-detection smart-segments build-annotations
.venv/bin/sponsor-detection smart-segments build-benchmark      # reserve channels first
.venv/bin/sponsor-detection data build-scriptsmith-dataset --config config/scriptsmith_smart_segments.toml
.venv/bin/sponsor-detection smart-segments freeze-gates

# Training and export (GPU; uses the conda env `llm`)
PYTHONPATH=src /home/curious/miniconda3/envs/llm/bin/python -m sponsor_detection \
  train-smart-segments --config config/train_smart_segments.toml
PYTHONPATH=src /home/curious/miniconda3/envs/llm/bin/python -m sponsor_detection \
  export-smart-segments --config config/export_smart_segments.toml
PYTHONPATH=src /home/curious/miniconda3/envs/llm/bin/python -m sponsor_detection \
  smart-segments evaluate-videos --config config/evaluate_smart_segments_videos.toml

# Tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 185 tests, 1 skipped
```

---

## 13. Artifact index

| Artifact | Path |
| --- | --- |
| Raw data audit | `reports/smart_segments_raw_data_audit.json` |
| Trainability audit | `reports/smart_segments_trainability_audit.json` |
| Canonical annotations manifest | `reports/smart_segment_annotations_manifest.json` |
| Frozen dataset manifest | `reports/smart_segments_windows_manifest.json` |
| Benchmark review manifest | `reports/smart_segments_benchmark_review_manifest.json` |
| Gate contract | `reports/smart_segments_gate_contract.json` |
| Training metrics | `reports/smart_segments_ettin_17m_metrics.json` |
| Export report / manifest | `reports/smart_segments_export_report.json`, `reports/smart_segments_export_manifest.json` |
| Video evaluation | `reports/smart_segments_video_evaluation.json` |
| ONNX bundle | `artifacts/smart_segments/` (gitignored) |
| Checkpoints | `checkpoints/smart_segments_ettin_17m/` (gitignored) |

## 14. Commit index (Flow-SponsorML, `origin/main`)

```
02784dea feat(ml): evaluate the multi-head model on held-out videos
62905f4f feat(ml): export the multi-head model to a versioned ONNX bundle
6394cc23 chore(ml): record the first multi-head smart segment training run
78966ba0 fix(ml): stop reporting undefined token precision in multi-head evals
c9efca3b chore(ml): rebuild frozen splits with benchmark channels excluded
63875c4d feat(ml): add the multi-head smart segment classifier and trainer
04111b10 chore(ml): record the smart segments benchmark review queue
00100eb6 fix(ml): reserve benchmark channels before building training splits
ca52dcd5 feat(ml): sample and freeze a multicategory benchmark
23c5ec1a chore(ml): record the frozen smart segments window dataset
9bb0c591 feat(ml): add per-category smart segment evaluation metrics
37df7b9c feat(ml): build per-category BILOU targets with loss masks
74949f25 feat(ml): add the versioned smart segment bundle manifest contract
c6029618 feat(ml): decode independent BILOU heads per category
826dad85 feat(ml): freeze the smart segment release gates
52eb8c08 feat(ml): record per-category supervision in the window dataset
ba39fdde feat(ml): add the smart segment supervision contract
cba6295d refactor(ml): consume canonical annotations in the dataset builders
e37769b8 feat(ml): add canonical smart segment annotation layer
c3bff80e chore(ml): record smart segments multicategory audit reports
```

Flow repo (Milestone 0): merge commit `ce9c2e2c` on `sync/upstream-20260929`;
tag `pre-upstream-merge-20260929`.
