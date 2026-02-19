#!/usr/bin/env python3
"""
Generate paper-ready Experiment A figures/tables from final_expA_compares JSON logs.

Outputs:
  - Figure 1: threshold-severity reversal frequency
  - Figure 2: B_c time series (subplot + overlay)
  - Figure 3: temporal persistence bar chart
  - Table 1: average metrics vs accuracy
  - Table 2: tail metrics vs accuracy
  - Figure A1: null-hypothesis comparison (cosine domain)
  - Figure A2: B_c histogram overlay
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Prevent matplotlib cache warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


THRESHOLDS = [1.0, 1.05, 1.1, 1.2]
SEGMENTS = ["early", "mid", "late"]

# Approximate client-side parameter dimension for ResNet18 CIFAR-style split at layer2.
# (Used for theoretical null N(0, 1/d) of cosine between random unit vectors.)
CLIENT_DIM_ESTIMATE = 675_392


@dataclass
class MethodRun:
    name: str
    file_path: Path
    b_c: np.ndarray
    a_c_ratio: np.ndarray
    g_drift: np.ndarray
    accuracy: np.ndarray


METHOD_ORDER = [
    "SFL IID",
    "USFL non-IID",
    "GAS non-IID",
    "MultiSFL non-IID",
    "MIX2SFL non-IID",
    "SFL non-IID",
]

METHOD_COLORS = {
    "SFL IID": "#2ca02c",
    "USFL non-IID": "#1f77b4",
    "GAS non-IID": "#9467bd",
    "MultiSFL non-IID": "#17becf",
    "MIX2SFL non-IID": "#ff7f0e",
    "SFL non-IID": "#d62728",
}


def _method_name_from_file(path: Path) -> Optional[str]:
    n = path.name
    if "result-usfl-" in n:
        return "USFL non-IID"
    if "result-sfl-" in n and "dist-uniform" in n:
        return "SFL IID"
    if "result-sfl-" in n and "shard_dirichlet" in n:
        return "SFL non-IID"
    if "result-mix2sfl-" in n:
        return "MIX2SFL non-IID"
    if n.startswith("results_gas_"):
        return "GAS non-IID"
    if n.startswith("results_multisfl_"):
        return "MultiSFL non-IID"
    return None


def _clean_numeric(values: List[object]) -> np.ndarray:
    out: List[float] = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        out.append(x)
    return np.asarray(out, dtype=np.float64)


def _extract_accuracy(payload: dict) -> np.ndarray:
    # SFL-style: payload["metric"][round] -> event list with MODEL_EVALUATED
    metric = payload.get("metric")
    if isinstance(metric, dict):
        rounds: List[int] = []
        for k in metric.keys():
            try:
                rounds.append(int(k))
            except (TypeError, ValueError):
                continue
        rounds.sort()
        acc: List[float] = []
        for r in rounds:
            events = metric.get(str(r), metric.get(r))
            if not isinstance(events, list):
                continue
            found: Optional[float] = None
            for e in events:
                if not isinstance(e, dict):
                    continue
                if e.get("event") != "MODEL_EVALUATED":
                    continue
                params = e.get("params", {})
                if not isinstance(params, dict) or "accuracy" not in params:
                    continue
                try:
                    found = float(params["accuracy"])
                except (TypeError, ValueError):
                    found = None
            if found is not None and not math.isnan(found):
                acc.append(found)
        return np.asarray(acc, dtype=np.float64)

    # GAS-style
    if isinstance(payload.get("accuracy"), list):
        return _clean_numeric(payload["accuracy"])

    # MultiSFL-style
    rounds_blob = payload.get("rounds")
    if isinstance(rounds_blob, list):
        acc: List[float] = []
        for row in rounds_blob:
            if isinstance(row, dict) and "accuracy" in row:
                try:
                    x = float(row["accuracy"])
                except (TypeError, ValueError):
                    continue
                if not math.isnan(x):
                    acc.append(x)
        return np.asarray(acc, dtype=np.float64)

    return np.asarray([], dtype=np.float64)


def _choose_latest(paths: List[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def load_runs(data_dir: Path) -> Dict[str, MethodRun]:
    per_method: Dict[str, List[Path]] = {}
    for p in sorted(data_dir.glob("*.json")):
        m = _method_name_from_file(p)
        if m is None:
            continue
        per_method.setdefault(m, []).append(p)

    runs: Dict[str, MethodRun] = {}
    for method_name, paths in per_method.items():
        chosen = _choose_latest(paths)
        with chosen.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        drift_history = payload.get("drift_history", {})
        if not isinstance(drift_history, dict):
            drift_history = {}

        b_c = _clean_numeric(drift_history.get("expA_B_c", []))
        a_c_ratio = _clean_numeric(drift_history.get("expA_A_c_ratio", []))
        g_drift = _clean_numeric(drift_history.get("G_drift", []))
        accuracy = _extract_accuracy(payload)

        runs[method_name] = MethodRun(
            name=method_name,
            file_path=chosen,
            b_c=b_c,
            a_c_ratio=a_c_ratio,
            g_drift=g_drift,
            accuracy=accuracy,
        )
    return runs


def _segment_slices(n: int) -> Dict[str, slice]:
    # Use fixed paper bins: rounds 1-100, 101-200, 201-300.
    # If a run has fewer than 300 rounds, the last bin is truncated to available rounds.
    return {
        "early": slice(0, min(100, n)),
        "mid": slice(min(100, n), min(200, n)),
        "late": slice(min(200, n), min(300, n)),
    }


def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_max(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def _safe_pct(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def _safe_freq(arr: np.ndarray, threshold: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > threshold))


def make_table_rows(runs: Dict[str, MethodRun]) -> Dict[str, List[dict]]:
    table1: List[dict] = []
    table2: List[dict] = []
    threshold_rows: List[dict] = []
    temporal_rows: List[dict] = []

    for method in METHOD_ORDER:
        if method not in runs:
            continue
        run = runs[method]
        b_c = run.b_c
        acc = run.accuracy

        acc_best = _safe_max(acc)
        acc_last50 = _safe_mean(acc[-50:]) if acc.size >= 50 else _safe_mean(acc)

        table1.append(
            {
                "Method": method,
                "Accuracy_best": acc_best,
                "Accuracy_last50": acc_last50,
                "avg_B_c": _safe_mean(b_c),
                "avg_A_c_ratio": _safe_mean(run.a_c_ratio),
                "avg_G_drift": _safe_mean(run.g_drift),
                "Source_JSON": str(run.file_path),
            }
        )
        table2.append(
            {
                "Method": method,
                "Accuracy_best": acc_best,
                "Accuracy_last50": acc_last50,
                "p90_B_c": _safe_pct(b_c, 90),
                "p95_B_c": _safe_pct(b_c, 95),
                "max_B_c": _safe_max(b_c),
                "freq_B_c_gt_1_1": _safe_freq(b_c, 1.1),
                "Source_JSON": str(run.file_path),
            }
        )

        for t in THRESHOLDS:
            threshold_rows.append(
                {
                    "Method": method,
                    "Threshold": t,
                    "Frequency": _safe_freq(b_c, t),
                }
            )

        segs = _segment_slices(b_c.size)
        for seg_name in SEGMENTS:
            seg = b_c[segs[seg_name]]
            temporal_rows.append(
                {
                    "Method": method,
                    "Segment": seg_name,
                    "Frequency_B_c_gt_1_05": _safe_freq(seg, 1.05),
                    "Frequency_B_c_gt_1_1": _safe_freq(seg, 1.1),
                }
            )

    return {
        "table1": table1,
        "table2": table2,
        "threshold": threshold_rows,
        "temporal": temporal_rows,
    }


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: List[dict], path: Path, float_fmt: str = "{:.4f}") -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")
    for row in rows:
        vals = []
        for k in keys:
            v = row[k]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("NaN")
                else:
                    vals.append(float_fmt.format(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figure1_threshold(rows: List[dict], out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.0), dpi=180)
    for method in METHOD_ORDER:
        xs = [r["Threshold"] for r in rows if r["Method"] == method]
        ys = [r["Frequency"] * 100.0 for r in rows if r["Method"] == method]
        if not xs:
            continue
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=method,
            color=METHOD_COLORS.get(method),
        )
    plt.xlabel("B_c Threshold")
    plt.ylabel("Frequency (%)")
    plt.title("Figure 1. Severe Alignment Reversal Frequency by Threshold")
    plt.xticks(THRESHOLDS, [f"{t:.2f}" for t in THRESHOLDS])
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(fontsize=8, ncols=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_figure2_timeseries(runs: Dict[str, MethodRun], out_subplot: Path, out_overlay: Path) -> None:
    # Subplot view
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=180, sharex=False, sharey=True)
    axes = axes.flatten()
    for idx, method in enumerate(METHOD_ORDER):
        ax = axes[idx]
        if method not in runs:
            ax.set_visible(False)
            continue
        arr = runs[method].b_c
        rounds = np.arange(1, arr.size + 1)
        ax.plot(rounds, arr, color=METHOD_COLORS.get(method), linewidth=1.2)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.axhline(1.1, color="#b22222", linestyle=":", linewidth=1.0)
        ax.set_title(f"{method}\nmax={np.max(arr):.3f}, p95={np.percentile(arr,95):.3f}", fontsize=9)
        ax.set_xlabel("Round")
        ax.set_ylabel("B_c")
        ax.grid(alpha=0.2)
        ax.set_ylim(0.85, 1.60)
    fig.suptitle("Figure 2. B_c Time Series by Method", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_subplot)
    plt.close(fig)

    # Overlay view
    plt.figure(figsize=(9.5, 5.0), dpi=180)
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        arr = runs[method].b_c
        rounds = np.arange(1, arr.size + 1)
        plt.plot(
            rounds,
            arr,
            linewidth=1.0,
            alpha=0.9,
            color=METHOD_COLORS.get(method),
            label=method,
        )
    plt.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0, label="B_c=1.0")
    plt.axhline(1.1, color="#b22222", linestyle=":", linewidth=1.0, label="B_c=1.1")
    plt.xlabel("Round")
    plt.ylabel("B_c")
    plt.title("Figure 2 (overlay). B_c Time Series")
    plt.grid(alpha=0.2)
    plt.ylim(0.85, 1.60)
    plt.legend(fontsize=8, ncols=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_overlay)
    plt.close()


def plot_figure3_temporal(rows: List[dict], out_path: Path) -> None:
    width = 0.24
    x = np.arange(len(METHOD_ORDER))

    seg_to_vals = {seg: [] for seg in SEGMENTS}
    for method in METHOD_ORDER:
        for seg in SEGMENTS:
            matched = [
                r["Frequency_B_c_gt_1_05"]
                for r in rows
                if r["Method"] == method and r["Segment"] == seg
            ]
            seg_to_vals[seg].append((matched[0] if matched else float("nan")) * 100.0)

    plt.figure(figsize=(10, 5), dpi=180)
    plt.bar(x - width, seg_to_vals["early"], width=width, label="Early (1-100)")
    plt.bar(x, seg_to_vals["mid"], width=width, label="Mid (101-200)")
    plt.bar(x + width, seg_to_vals["late"], width=width, label="Late (201-300)")
    plt.xticks(x, METHOD_ORDER, rotation=20, ha="right")
    plt.ylabel("Frequency of B_c > 1.05 (%)")
    plt.title("Figure 3. Temporal Persistence of Severe Reversal")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_figure_a2_histogram(runs: Dict[str, MethodRun], out_path: Path) -> None:
    plt.figure(figsize=(9, 5), dpi=180)
    bins = np.linspace(0.85, 1.60, 80)
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        plt.hist(
            runs[method].b_c,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            alpha=0.95,
            color=METHOD_COLORS.get(method),
            label=method,
        )
    plt.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    plt.axvline(1.1, color="#b22222", linestyle=":", linewidth=1.0)
    plt.xlabel("B_c")
    plt.ylabel("Density")
    plt.title("Figure A2. B_c Distribution (Histogram Overlay)")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8, ncols=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _normal_pdf(x: np.ndarray, sd: float) -> np.ndarray:
    coeff = 1.0 / (sd * math.sqrt(2.0 * math.pi))
    return coeff * np.exp(-0.5 * (x / sd) ** 2)


def plot_figure_a1_null(runs: Dict[str, MethodRun], out_path: Path) -> None:
    cos_data: Dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        if method in runs:
            cos_data[method] = 1.0 - runs[method].b_c

    sd = 1.0 / math.sqrt(float(CLIENT_DIM_ESTIMATE))
    x_pdf = np.linspace(-0.25, 0.05, 1000)
    y_pdf = _normal_pdf(x_pdf, sd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)

    # Left: cosine distribution vs theoretical null
    ax = axes[0]
    bins = np.linspace(-0.25, 0.05, 90)
    focus_methods = ["SFL IID", "USFL non-IID", "SFL non-IID", "MIX2SFL non-IID"]
    for method in focus_methods:
        if method not in cos_data:
            continue
        ax.hist(
            cos_data[method],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=METHOD_COLORS.get(method),
            label=method,
        )
    ax.plot(x_pdf, y_pdf, color="black", linewidth=1.2, linestyle="--", label="Theoretical N(0,1/d)")
    ax.set_yscale("log")
    ax.set_xlabel("cos(μ_c, c_c) = 1 - B_c")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("A1-left. Empirical Cosine vs High-D Null")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, frameon=False)

    # Right: tail survival P(cos <= -t)
    ax = axes[1]
    t_grid = np.array([0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    # Theoretical survival for lower tail
    # P(X <= -t) = 0.5 * erfc(t/(sd*sqrt(2)))
    theo = 0.5 * np.vectorize(math.erfc)(t_grid / (sd * math.sqrt(2.0)))
    ax.plot(t_grid, theo, color="black", linestyle="--", linewidth=1.2, label="Theoretical null")
    for method in focus_methods:
        if method not in cos_data:
            continue
        c = cos_data[method]
        surv = np.array([np.mean(c <= -t) for t in t_grid], dtype=np.float64)
        ax.plot(t_grid, surv, marker="o", linewidth=1.2, markersize=3, color=METHOD_COLORS.get(method), label=method)
    ax.set_yscale("log")
    ax.set_xlabel("t  (tail threshold on cos <= -t)")
    ax.set_ylabel("Tail probability (log scale)")
    ax.set_title("A1-right. Empirical Tail vs Null Tail")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        f"Figure A1. Null Hypothesis Check (d={CLIENT_DIM_ESTIMATE:,}, sd={sd:.4e})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_manifest(runs: Dict[str, MethodRun], out_path: Path) -> None:
    payload = {}
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        run = runs[method]
        payload[method] = {
            "source_json": str(run.file_path),
            "n_rounds_bc": int(run.b_c.size),
            "n_rounds_accuracy": int(run.accuracy.size),
            "acc_best": _safe_max(run.accuracy),
            "acc_last50": _safe_mean(run.accuracy[-50:]) if run.accuracy.size >= 50 else _safe_mean(run.accuracy),
            "avg_B_c": _safe_mean(run.b_c),
            "avg_A_c_ratio": _safe_mean(run.a_c_ratio),
            "avg_G_drift": _safe_mean(run.g_drift),
            "p90_B_c": _safe_pct(run.b_c, 90),
            "p95_B_c": _safe_pct(run.b_c, 95),
            "max_B_c": _safe_max(run.b_c),
            "freq_B_c_gt_1_1": _safe_freq(run.b_c, 1.1),
        }
    payload["_null_hypothesis"] = {
        "client_dim_estimate": CLIENT_DIM_ESTIMATE,
        "cosine_std_theoretical": 1.0 / math.sqrt(float(CLIENT_DIM_ESTIMATE)),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("final_expA_compares"),
        help="Directory containing raw result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final_expA_compares/paper_assets"),
        help="Directory to store generated figure/table assets.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.data_dir)
    missing = [m for m in METHOD_ORDER if m not in runs]
    if missing:
        print(f"[WARN] Missing methods: {missing}")

    rows = make_table_rows(runs)

    # CSV/Markdown tables
    write_csv(rows["table1"], args.output_dir / "table1_average_vs_accuracy.csv")
    write_csv(rows["table2"], args.output_dir / "table2_tail_vs_accuracy.csv")
    write_csv(rows["threshold"], args.output_dir / "figure1_threshold_curve_data.csv")
    write_csv(rows["temporal"], args.output_dir / "figure3_temporal_data.csv")

    write_markdown_table(
        rows["table1"],
        args.output_dir / "table1_average_vs_accuracy.md",
    )
    write_markdown_table(
        rows["table2"],
        args.output_dir / "table2_tail_vs_accuracy.md",
    )

    # Figures
    plot_figure1_threshold(rows["threshold"], args.output_dir / "figure1_threshold_curve.png")
    plot_figure2_timeseries(
        runs,
        args.output_dir / "figure2_bc_timeseries_subplots.png",
        args.output_dir / "figure2_bc_timeseries_overlay.png",
    )
    plot_figure3_temporal(rows["temporal"], args.output_dir / "figure3_temporal_persistence.png")
    plot_figure_a2_histogram(runs, args.output_dir / "figureA2_bc_hist_overlay.png")
    plot_figure_a1_null(runs, args.output_dir / "figureA1_null_hypothesis.png")

    write_manifest(runs, args.output_dir / "analysis_manifest.json")

    print("[OK] Generated paper assets:")
    for p in sorted(args.output_dir.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
