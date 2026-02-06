# experiment_core

`experiment_core` provides a single experiment spec and a single normalized result format for:

- `sfl_framework-fork-feature-training-tracker` (`framework: sfl`)
- `GAS_implementation` (`framework: gas`)
- `multisfl_implementation` (`framework: multisfl`)

## Goals

- One spec file for config instead of per-framework ad-hoc flags/env.
- One normalized output JSON for downstream analysis.
- Keep each framework's training loop intact (adapter-based integration).

## Run

From repo root:

```bash
python -m experiment_core.run_experiment \
  --spec experiment_core/examples/sfl_iid_ref.json \
  --repo-root .
```

Normalize an existing raw result only:

```bash
python -m experiment_core.run_experiment \
  --spec experiment_core/examples/normalize_only.json \
  --repo-root .
```

## Spec shape

Top-level keys:

- `framework`: `sfl | gas | multisfl`
- `method`: free-form method name (`sfl`, `usfl`, `mix2sfl`, ...)
- `common`: normalized common experiment fields
- `execution`: run/normalize mode, paths, env, command override
- `framework_overrides`: adapter-specific options

### Important `execution` options

- `mode`: `run` or `normalize_only`
- `python`: python executable (default `python`)
- `cwd`: optional working directory override
- `command`: optional full command override (list)
- `raw_result_path`: explicit raw result path
- `raw_result_glob`: auto-discovery glob pattern
- `normalized_output`: normalized output path

## Normalized output

Each normalized file includes:

- `schema_version`
- `framework`
- `run_meta` (`command`, `cwd`, timestamps, exit code)
- `raw_result_path`
- `config`
- `accuracy_by_round`
- `drift_history`
- `alignment_history`

Alignment fields:

- SFL/GAS: `A_cos`, `M_norm`, `n_valid_alignment`
- MultiSFL: `A_cos_client`, `M_norm_client`, `A_cos_server`, `M_norm_server`

## Current limits

- GAS partition mode is still largely controlled inside `GAS_main.py` defaults.
  Adapter currently maps core env overrides but does not fully rewrite GAS partition logic.
- Adapter-level command mapping covers common parameters; anything custom should be passed via
  `framework_overrides` or `execution.command`.

## Design choice

This is intentionally non-invasive: it standardizes input/output first, then allows deeper
framework internals refactor later without breaking experiment scripts.
