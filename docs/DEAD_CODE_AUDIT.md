# Dead Code Audit Report

**Date:** 2026-02-07
**Branch:** `codex/experiment-core-refactor`
**Scope:** All 3 framework directories

---

## Overview

Automated audit of unused files, dead functions, unreachable code paths, and legacy artifacts
across the three framework implementations. Each item is rated by confidence:

- **HIGH** — Zero references found anywhere in the codebase
- **MEDIUM** — Reachable only through non-default/legacy config paths
- **LOW** — Redundant but harmless (duplicate conditions, unused parameters)

### Summary

| Framework | Dead Files | Dead Functions/Classes | Dead Code Paths |
|-----------|-----------|----------------------|-----------------|
| SFL | 13+ files (analyzer module, client models, artifacts) | 3 | 2 |
| GAS | 9 files (split dirs, workload script) | 6 | 4 |
| MultiSFL | 2 files (stub, scaffolding) | 10+ | 0 |

---

## SFL Framework

**Directory:** `sfl_framework-fork-feature-training-tracker/`

### Unused Files

#### Analyzer Module (7 files) — HIGH

`server/modules/trainer/analyzer/` is a gradient analysis module that was implemented but
never wired into the training pipeline. `get_analyzer()` is never called by any stage
organizer, handler, or entry point.

```
server/modules/trainer/analyzer/analyzer.py          # Analyzer class, get_analyzer() factory
server/modules/trainer/analyzer/base_analyzer.py     # BaseAnalyzer
server/modules/trainer/analyzer/CKA_analyzer.py      # CKA analysis
server/modules/trainer/analyzer/conflict_ratio_analyzer.py
server/modules/trainer/analyzer/cosine_similarity_analyzer.py
server/modules/trainer/analyzer/L2_analyzer.py
server/modules/trainer/analyzer/mean_variance_analyzer.py
```

#### Client Custom Model Files (6 files) — HIGH

`client/modules/model/custom_model/` contains model definitions that are never imported.
The client side doesn't have a model factory — models are received from the server.
The server-side counterparts in `server/modules/model/custom_model/` ARE properly used.

```
client/modules/model/custom_model/alexnet_1ch.py     # Only for unreachable alexnet_legacy
client/modules/model/custom_model/alexnet_cifar.py
client/modules/model/custom_model/alexnet_mnist.py
client/modules/model/custom_model/alexnet_scala.py
client/modules/model/custom_model/lenet.py
client/modules/model/custom_model/tiny_vgg11.py
```

#### Standalone Scripts (3 files) — MEDIUM

These are development/analysis tools. Never imported by the framework, but were
intentionally standalone `__main__` scripts.

| File | Description |
|------|-------------|
| `cal.py` | Batch scheduling algorithm prototype (Korean comments) |
| `test.py` | Accuracy diff percentage calculator with hardcoded values |
| `utils/diff.py` | Layer-wise model parameter comparison and swap-evaluate tool |

#### Result Artifacts (4 files) — HIGH

Old experiment output JSONs from Jan 25 at the framework root. Already excluded by
`.gitignore` (`result-*.json` pattern in framework's `.gitignore`).

```
result-usfl-resnet18_flex-cifar10-...-20260125-142956.json
result-usfl-resnet18_flex-cifar10-...-20260125-144946.json
result-usfl-resnet18_flex-cifar10-...-20260125-152810.json
result-usfl-resnet18_flex-cifar10-...-20260125-153359.json
```

#### `emulation.py` — MEDIUM (standalone entry point, NOT dead)

Alternative entry point for real-process experiments (server/client as separate
subprocesses via Popen). Independent of `simulation.py`. Not dead — just an
alternate workflow that is rarely used.

### Dead Functions

| Location | Function | Issue | Confidence |
|----------|----------|-------|------------|
| `server/main.py:316` | `simulation_main()` | Never called. Body identical to `main()` at line 310. | HIGH |
| `server/main.py:238` | `_validate_scala_parameters()` (first definition) | Overwritten by second definition at line 252. Split_ratio validation in lines 238-249 is dead. | HIGH |
| `server/modules/model/alexnet.py:24` | `alexnet_legacy` branch | Unreachable: `get_model()` only routes `"alexnet"` and `"alexnet_scala"` to AlexNet class. Not in valid model choices in `server_args.py`. | MEDIUM |

### Dead Configuration

| Item | Details | Confidence |
|------|---------|------------|
| `sfl-u` method | Referenced in `server_args.py` (choices), `client_args.py` (docs), `handler.py` (routes to SFLHandler). But **no stage organizer mapping** in `get_stage_organizer()` — would crash with `ValueError`. | MEDIUM |
| `resnet18_flex` missing from validation | `_validate_workload()` in `server/main.py` does not list `resnet18_flex` as valid. Works via `simulation.py` (skips validation) but fails via `server/main.py`. | LOW |

---

## GAS Implementation

**Directory:** `GAS_implementation/`

### Unused Files

#### Legacy Split Directories (8 files) — HIGH

Never imported anywhere. Replaced by dynamic splitting in `network.py`
(`_build_alexnet_split_models()`, `_resolve_alexnet_split_index()`).

```
utils/AlexNet_Split/L1.py
utils/AlexNet_Split/L2.py
utils/AlexNet_Split/L3.py
utils/AlexNet_Split/L4.py
utils/VGG16_split/L1.py
utils/VGG16_split/L2.py
utils/VGG16_split/L3.py
utils/VGG16_split/L4.py
```

#### Standalone Script — HIGH

| File | Description |
|------|-------------|
| `utils/obtain_workload.py` | FLOPs/parameter counter using `fvcore`. Contains duplicate model class definitions. Never imported. |

### Dead Functions/Classes

| Location | Item | Issue | Confidence |
|----------|------|-------|------------|
| `g_measurement.py:24` | `GradientCollector` class | Only used by dead `measure_and_record()` | HIGH |
| `g_measurement.py:562` | `measure_and_record()` | Never called from `GAS_main.py` — replaced by inline logic in `finalize_g_measurement()` | HIGH |
| `g_measurement.py:356` | `collect_current_gradients()` | Only caller is dead `measure_and_record()` | HIGH |
| `g_measurement.py:478` | `compute_all_g_scores()` | Only caller is dead `measure_and_record()` | HIGH |
| `network.py:209` | `inversion_model()` + `custom_AE` class | Model inversion attack remnant. Never called. | HIGH |
| `dataset.py:71` | `CustomDataset` class | Never instantiated. Contains debug `print()` in `__getitem__`. | HIGH |

### Dead Code Paths

| Location | Item | Confidence |
|----------|------|------------|
| `network.py` various | `return_features` param in AlexNet forward — never passed as `True` | HIGH |
| `network.py:91,123-187` | `twoLogit` param paths — GAS_main.py never passes `True` | HIGH |
| `network.py:644-756` | `ResNet18DownCifar/UpCifar` — only reachable with `split_ratio="half"` AND `split_layer=None` AND `resnet_image_style=False` (none are default) | HIGH |
| `network.py:759-871` | `ResNet18DownCifarLight/UpCifarHeavy` — same, with `split_ratio="quarter"` | HIGH |
| `GAS_main.py:248-259` | Commented-out hardcoded `clients_computing`, `clients_position`, `rates` arrays | LOW |

---

## MultiSFL Implementation

**Directory:** `multisfl_implementation/`

### Unused Files

| File | Issue | Confidence |
|------|-------|------------|
| `main.py` | Stub file: just `print("Hello from multisfl-framework!")`. Auto-generated placeholder. Real entry point is `run_multisfl.py`. | HIGH |
| `pyproject.toml` | Minimal stub: name + version only. No deps, no build system, no entry points. Unused by any build or deploy process. | MEDIUM |

### Dead Functions/Classes

| Location | Item | Confidence |
|----------|------|------------|
| `client.py:204` | `Client.get_local_class_counts()` — never called | HIGH |
| `drift_measurement.py:530` | `MultiSFLDriftTracker.clear()` — never called | HIGH |
| `drift_measurement.py:120` | `BranchDriftState.reset()` — never called | HIGH |
| `g_measurement.py:501` | `GMeasurementSystem.start_accumulated_round()` — never wired into trainer | HIGH |
| `g_measurement.py:511` | `GMeasurementSystem.accumulate_client_gradient()` — same | HIGH |
| `g_measurement.py:523` | `GMeasurementSystem.accumulate_server_gradient()` — same | HIGH |
| `g_measurement.py:532` | `GMeasurementSystem.finalize_accumulated_round()` — same | HIGH |
| `g_measurement.py:413` | `GradientAccumulator.reset()` — only caller is dead code above | MEDIUM |
| `models.py:185-279` | 6 legacy AlexNet split classes (`AlexNetDownCifar`, `AlexNetUpCifar`, `AlexNetDownCifarLight`, `AlexNetUpCifarHeavy`, `AlexNetDown`, `AlexNetUp`) — bypassed by dynamic `_build_alexnet_split_models()` | HIGH |
| `models.py:1357` | `MODEL_INFO` dict — informational, never referenced | HIGH |

### Unused Imports

| Location | Import | Confidence |
|----------|--------|------------|
| `run_multisfl.py:5` | `from torch.utils.data import DataLoader` | MEDIUM |
| `run_multisfl.py:7` | `import time` | MEDIUM |

---

## Common Patterns

The same types of dead code appear across all three frameworks:

1. **Legacy hardcoded model splits** — All frameworks originally used fixed split classes
   (e.g., `AlexNetDownCifar`, `ResNet18DownCifarLight`). These were replaced by dynamic
   splitting functions that build split models programmatically. The old classes remain.

2. **Alternative API surfaces** — Code was written for multiple approaches, but only one
   was wired into the actual training loop:
   - SFL: analyzer module (7 files, never connected to stage organizers)
   - GAS: `measure_and_record()` path (replaced by inline `finalize_g_measurement()`)
   - MultiSFL: streaming accumulation API (replaced by batch `measure_round()`)

3. **Standalone scratch scripts** — Research prototype code (`cal.py`, `test.py`,
   `obtain_workload.py`) that was useful during development but never integrated.

---

## Cleanup Priority

**Tier 1 — Safe to delete immediately (HIGH confidence, zero risk):**
- SFL: analyzer/ (7 files), client custom_model/ (6 files), result artifacts (4 files)
- GAS: AlexNet_Split/ (4 files), VGG16_split/ (4 files), obtain_workload.py
- MultiSFL: main.py, pyproject.toml

**Tier 2 — Dead functions (HIGH confidence, minor risk):**
- Removing unused functions/classes within files. Low risk but requires careful editing
  to avoid breaking any undiscovered references.

**Tier 3 — Review needed (MEDIUM confidence):**
- Standalone scripts (cal.py, test.py, diff.py) — may have sentimental/reference value
- Dead config paths (sfl-u method, alexnet_legacy) — removing references across
  multiple files
- Dead code paths in network.py (twoLogit, return_features) — parameter removal
  affects function signatures
