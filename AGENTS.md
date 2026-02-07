# Repository Guidelines

## Project Structure & Module Organization
- `sfl_framework-fork-feature-training-tracker/`: unified SFL/USFL framework. Entry points: `simulation.py` (simulation), `emulation.py` (process mode). Server/client modules live in `server/` and `client/`.
- `GAS_implementation/`: standalone GAS method; `GAS_main.py` plus `utils/` for models, data, and metrics.
- `multisfl_implementation/`: MultiSFL experiments; `run_multisfl.py`, `exp.sh`, and the `multisfl/` package.
- `experiment_core/`: unified experiment framework — adapters, spec generation, batch runner, result normalization.
- `experiment_configs/`: shared `common.json` + per-method config JSONs.
- `shared/`: cross-framework utilities (update alignment metrics).
- `docs/`: documentation (G measurement guide, SCAFFOLD usage, alignment docs).
- `deploy.sh`: multi-GPU server deployment automation.
- `deploy/`: deployment supporting files (`remote_run.sh`, `setup_rclone_gdrive.sh`, `deploy_servers.json`).

## Build, Test, and Development Commands
```bash
cd sfl_framework-fork-feature-training-tracker
pip install -r requirements.txt
python simulation.py        # SFL/USFL; edit configs inside the file
python result_parser.py     # summarize experiment logs
```

```bash
cd sfl_framework-fork-feature-training-tracker
python server/main.py [args]   # emulation server (separate terminal)
python client/main.py [args]   # emulation client(s)
```

```bash
cd GAS_implementation && python GAS_main.py
cd multisfl_implementation && python run_multisfl.py [args]
./exp.sh  # MultiSFL batch runs
```

### Deployment
```bash
./deploy.sh run usfl@server-a:0 gas@server-b:1   # distributed execution
./deploy.sh status                                 # check all servers
./deploy.sh collect --local                        # collect results
```

## Coding Style & Naming Conventions
- Python with 4-space indentation; keep changes aligned with surrounding style.
- `snake_case` for functions/variables, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- No repo-wide formatter or linter config found; avoid sweeping reformat-only changes.

## Testing Guidelines
- No automated unit-test framework detected; validation is typically via experiment runs.
- For quick sanity checks, `sfl_framework-fork-feature-training-tracker/test.py` runs a small analysis script.
- When modifying training logic, record the config/seed used and compare a small run before/after.

## Commit & Pull Request Guidelines
- Recent history favors Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`), but some commits are free-form; use a short, imperative subject line and keep to one topic.
- PRs should state which track(s) are affected, include key config flags (e.g., `-M`, `-distr`, `-sl`), and attach relevant logs or metric deltas. Link issues when applicable.

## Configuration & Experiment Notes
- Core config for the unified framework lives in `sfl_framework-fork-feature-training-tracker/server/server_args.py` and `sfl_framework-fork-feature-training-tracker/simulation.py`.
- Unified experiment configs: `experiment_configs/common.json` (shared) + per-method JSONs.
- Document changes to data distribution or split strategy (`-distr`, `-diri-alpha`, `-ss`, `-sl`) in your PR description.
