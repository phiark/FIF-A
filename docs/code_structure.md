# FIF-A Codebase Structure & Cleanup Roadmap

This document captures the current repository layout, the hygiene plan for
v1.0.4, and the concrete actions already taken to keep the code practical for
experimentation, optimization, and troubleshooting.

---

## 1. Directory Overview

| Path | Purpose | Notes |
| --- | --- | --- |
| `fif_mvp/` | Main training stack | Contains CLI, dataloaders, models, training loop, utilities. |
| `Project_Phase_Reports/` | Publication-ready artifacts | HTML + Markdown reports (phase summary, viz). |
| `docs/` | Experiment + design documentation | `experiment_design.md`, result reports, **this file**. |
| `scripts/` | Launch scripts + helpers | Split into `snli_*.sh`, `sst2_*`, misc utilities. |
| `result/` | Run artifacts (metrics, alerts, configs) | One directory per timestamped run. |
| `fif_simple/` | Legacy prototype | Keep for historical reference only; no code path depends on it. |

---

## 2. Refactor Goals (v1.0.4)

1. **Runtime bottlenecks** – eliminate Python hotspots in graph construction &
   per-step telemetry so Hybrid training is debuggable without profiling every run.
2. **Code discoverability** – tighten module contracts (e.g., dataloaders vs.
   trainer vs. models) and remove unused helpers that make energy flows harder to
   trace.
3. **Troubleshooting UX** – surface slow-path stats (step time, guard/watch events)
   in a single place and document how to reproduce any experiment.
4. **Legacy isolation** – keep `fif_simple` + latex artifacts clearly labelled as
   non-MVP code to avoid dead-file confusion.

---

## 3. Completed Cleanup Actions

| Area | Change | Impact |
| --- | --- | --- |
| Graph building | `build_knn_edges_batched` rewritten with batched matmul + masking | Drops the `B×L` Python loop, removing the largest remaining wall-time hotspot (tracked by T-018/T-019). |
| Dead code | Removed unused `FrictionLayer._build_edges` helper | Eliminates misleading API surface (no call sites since v1.0.0). |
| Docs | Added this roadmap to document directory layout + hygiene plan | Provides a canonical reference when onboarding or filing infra issues. |

---

## 4. Pending / Next Steps

1. **Graph caching**: optional disk-backed cache for window/knn edges to reduce
   rebuilds when `recompute_mu=True`.
2. **Energy signal integration**: wire alert stats (std/p90 trends) into tensorboard
   or CSV to short-circuit failures pre-train.
3. **Legacy archive**: relocate `fif_simple/` into `docs/archive/` once all references
   are captured inside documentation.
4. **Automated hygiene checks**: add `tox` job or CI step that runs `ruff` +
   selective unit tests to catch unused imports / dead paths before merging.

Track these items on the `WORK_BOARD` (v1.0.4 section) to preserve alignment with
version planning.
