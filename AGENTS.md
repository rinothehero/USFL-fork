# Repository Guidelines

## Project Structure & Module Organization
- `sfl_framework-fork-feature-training-tracker/`: unified SFL/USFL framework. Entry points: `simulation.py` (simulation), `emulation.py` (process mode). Server/client modules live in `server/` and `client/`.
- `GAS_implementation/`: standalone GAS method; `GAS_main.py` plus `utils/` for models, data, and metrics.
- `multisfl_implementation/`: MultiSFL experiments; `run_multisfl.py`, `exp.sh`, and the `multisfl/` package.
- Top-level helpers: `run_alexnet_cifar_gcheck.py`, `run_gas_multisfl_first3.py`.

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
- Document changes to data distribution or split strategy (`-distr`, `-diri-alpha`, `-ss`, `-sl`) in your PR description.
