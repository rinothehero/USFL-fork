#!/usr/bin/env python3
"""
Split Federated Learning Experiment Dashboard Generator

Usage:
    python sfl_dashboard.py <results_directory> [--output dashboard.html] [--open]

Examples:
    python sfl_dashboard.py ./results/
    python sfl_dashboard.py ./results/ --output my_dashboard.html
    python sfl_dashboard.py ./results/ --open   # 생성 후 브라우저에서 자동 열기

JSON 파일 형식:
    - SFL/USFL 형식: config, metric, drift_history, drift_measurements 포함
    - GAS 형식: config, accuracy, v_value, drift_history, time_record 포함
    - 디렉토리 내 모든 .json 파일을 자동 탐색합니다.

Dependencies:
    pip install plotly   # 없으면 CDN fallback (인터넷 필요)
"""

import json
import os
import sys
import argparse
import webbrowser
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# 1. JSON 파일 파싱
# ─────────────────────────────────────────────

def detect_experiment_type(data: dict) -> str | None:
    """JSON 데이터에서 실험 유형을 자동 감지"""
    # SFL/USFL: event-based metric system + drift_history
    if "metric" in data and "drift_history" in data:
        cfg = data.get("config", {})
        method = cfg.get("method", "").lower()
        if method in ("sfl", "usfl", "splitfed", "splitnn"):
            return method
        return "sfl"  # default for metric-based files
    # MultiSFL: rounds array with per-round accuracy
    if "rounds" in data and isinstance(data["rounds"], list) and len(data["rounds"]) > 0:
        if "accuracy" in data["rounds"][0]:
            return "multisfl"
    # GAS: top-level accuracy list
    if "accuracy" in data and isinstance(data["accuracy"], list):
        cfg = data.get("config", {})
        if "method" in cfg:
            return cfg["method"].lower().replace(" ", "_")
        return "gas"
    return None


def make_label(data: dict, filename: str, exp_type: str = "") -> str:
    """실험 설정에서 사람이 읽기 좋은 라벨 자동 생성"""
    cfg = data.get("config", {})
    method = cfg.get("method", "").upper()
    if not method and exp_type:
        method = exp_type.upper()
    if not method:
        method = "UNKNOWN"
    dist = cfg.get("distributer", cfg.get("partition", cfg.get("distribution", "")))
    alpha = cfg.get("dirichlet_alpha", cfg.get("alpha_dirichlet", cfg.get("alpha", "")))
    selector = cfg.get("selector", "")
    grad_shuf = cfg.get("gradient_shuffle", cfg.get("generate", False))
    bs = cfg.get("batch_size", "")

    parts = [method]

    # Distribution
    if "uniform" in str(dist).lower() or str(dist).lower() == "iid":
        parts.append("IID")
    elif dist:
        label_dist = "Non-IID"
        if alpha:
            label_dist += f" α={alpha}"
        parts.append(label_dist)

    # Special features
    if selector and selector not in ("uniform", "random"):
        parts.append(selector.upper())
    if grad_shuf and str(grad_shuf).lower() not in ("false", "none", "0"):
        parts.append("GradShuf")
    if bs:
        parts.append(f"bs={bs}")

    label = ", ".join(parts)

    # Deduplicate: if label is too generic, append filename hint
    if label == "UNKNOWN":
        label = Path(filename).stem[:40]

    return label


def extract_experiment(filepath: str) -> dict | None:
    """단일 JSON 파일에서 실험 데이터 추출"""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ⚠ Skipping {filepath}: {e}")
        return None

    exp_type = detect_experiment_type(data)
    if exp_type is None:
        print(f"  ⚠ Skipping {filepath}: not a recognized SFL/USFL/GAS result file")
        return None

    cfg = data.get("config", {})
    dh = data.get("drift_history", {})

    # --- Accuracy ---
    if "metric" in data:
        # SFL/USFL: extract from event system
        acc_data = []
        for rnd_key in sorted(data["metric"].keys(), key=int):
            for evt in data["metric"][rnd_key]:
                if evt.get("event") == "MODEL_EVALUATED":
                    acc = evt["params"].get("accuracy", evt["params"].get("acc"))
                    if acc is not None:
                        acc_data.append((int(rnd_key), float(acc)))
        acc_rounds = [d[0] for d in acc_data]
        acc_values = [d[1] for d in acc_data]
    elif "rounds" in data and isinstance(data["rounds"], list):
        # MultiSFL: extract from rounds array
        acc_rounds = [r["round"] for r in data["rounds"] if "accuracy" in r]
        acc_values = [float(r["accuracy"]) for r in data["rounds"] if "accuracy" in r]
    else:
        # GAS: top-level accuracy list
        acc_values = [float(v) for v in data.get("accuracy", [])]
        acc_rounds = list(range(1, len(acc_values) + 1))

    # --- Drift rounds ---
    per_round = dh.get("per_round", [])
    drift_rounds = [pr["round"] for pr in per_round] if per_round else list(range(1, len(next(
        (v for v in dh.values() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))),
        []
    )) + 1))

    # --- All numeric metrics from drift_history ---
    metrics = {}
    for k, v in dh.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            metrics[k] = v

    # --- MultiSFL per-round metrics (p_r, fgn_r, update norms, etc.) ---
    if "rounds" in data and isinstance(data["rounds"], list) and len(data["rounds"]) > 0:
        round_keys = [k for k in data["rounds"][0].keys() if k not in ("round", "accuracy")]
        for k in round_keys:
            vals = [r.get(k) for r in data["rounds"] if isinstance(r.get(k), (int, float))]
            if vals and k not in metrics:
                metrics[k] = vals

    # --- G Measurement metrics ---
    # GAS: g_history is a dict of time-series lists (directly usable)
    gh = data.get("g_history", {})
    for k, v in gh.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            gkey = f"g_{k}" if not k.startswith("g_") else k
            if gkey not in metrics:
                metrics[gkey] = v

    # SFL/MultiSFL: g_measurements is an array of event dicts
    # Extract scalar fields from params into time-series
    gm = data.get("g_measurements", [])
    g_meas_rounds = []
    if gm:
        # Collect all scalar keys from params
        g_scalar_keys = set()
        for entry in gm:
            params = entry.get("params", entry)
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    g_scalar_keys.add(k)
                elif isinstance(v, dict):
                    # Nested dicts like server: {G, G_rel, D_cosine}
                    for subk, subv in v.items():
                        if isinstance(subv, (int, float)):
                            g_scalar_keys.add(f"{k}_{subk}")
        # Extract actual measurement round numbers for accurate x-axis
        g_meas_rounds = []
        for entry in gm:
            rnd = entry.get("round")
            if rnd is not None:
                g_meas_rounds.append(int(rnd))

        # Build time-series
        for k in sorted(g_scalar_keys):
            vals = []
            for entry in gm:
                params = entry.get("params", entry)
                if k in params and isinstance(params[k], (int, float)):
                    vals.append(params[k])
                elif "_" in k:
                    # Try nested: "server_G" -> params["server"]["G"]
                    parts = k.split("_", 1)
                    nested = params.get(parts[0], {})
                    if isinstance(nested, dict) and parts[1] in nested:
                        vals.append(nested[parts[1]])
            if vals:
                gkey = f"g_{k}"
                if gkey not in metrics:
                    metrics[gkey] = vals

    # --- Extra fields ---
    v_values = data.get("v_value", [])
    time_record = data.get("time_record", [])

    label = make_label(data, os.path.basename(filepath), exp_type=exp_type)

    return {
        "label": label,
        "config": cfg,
        "acc_rounds": acc_rounds,
        "acc_values": acc_values,
        "drift_rounds": drift_rounds,
        "g_meas_rounds": g_meas_rounds,
        "metrics": metrics,
        "v_values": v_values,
        "time_record": time_record,
        "exp_type": exp_type,
        "filename": os.path.basename(filepath),
    }


def load_all_experiments(directory: str) -> dict:
    """디렉토리 내 모든 JSON 파일을 로드"""
    experiments = {}
    json_files = sorted(Path(directory).glob("*.json"))

    if not json_files:
        print(f"❌ No .json files found in {directory}")
        sys.exit(1)

    print(f"📂 Found {len(json_files)} JSON files in {directory}")

    label_counts: dict[str, int] = {}
    for fp in json_files:
        print(f"  → Loading {fp.name}...")
        exp = extract_experiment(str(fp))
        if exp:
            # Handle duplicate labels
            base_label = exp["label"]
            if base_label in label_counts:
                label_counts[base_label] += 1
                exp["label"] = f"{base_label} #{label_counts[base_label]}"
            else:
                label_counts[base_label] = 1
            experiments[exp["label"]] = exp

    print(f"✅ Loaded {len(experiments)} experiments")
    return experiments


# ─────────────────────────────────────────────
# 2. 메트릭 분석 & 차트 데이터 생성
# ─────────────────────────────────────────────

# 10 colors for up to 10 experiments
PALETTE = [
    "#6c9eff", "#ff6b8a", "#50e3c2", "#ffc85b",
    "#b388ff", "#ff8a65", "#4dd0e1", "#aed581",
    "#f48fb1", "#90a4ae",
]


def classify_metrics(experiments: dict) -> dict:
    """메트릭을 공통/실험별로 분류"""
    metric_presence: dict[str, list[str]] = {}
    for label, exp in experiments.items():
        for m in exp["metrics"]:
            metric_presence.setdefault(m, []).append(label)

    all_labels = set(experiments.keys())
    common = {m for m, labs in metric_presence.items() if set(labs) == all_labels}
    per_exp: dict[str, set[str]] = {}
    for m, labs in metric_presence.items():
        if m not in common:
            for lab in labs:
                per_exp.setdefault(lab, set()).add(m)

    return {"common": sorted(common), "per_experiment": per_exp, "all": metric_presence}


def build_chart_traces(experiments: dict) -> dict:
    """모든 차트의 traces + layout 생성"""
    labels = list(experiments.keys())
    charts = {}

    def layout(ytitle, tickfmt=None, yrange=None, xtitle="Round / Epoch"):
        l = {
            "paper_bgcolor": "transparent", "plot_bgcolor": "transparent",
            "font": {"color": "#e4e6f0", "family": "Inter, -apple-system, sans-serif", "size": 12},
            "margin": {"t": 10, "r": 20, "b": 50, "l": 65},
            "xaxis": {"title": {"text": xtitle, "standoff": 10}, "gridcolor": "#2d3148", "zerolinecolor": "#2d3148", "tickfont": {"size": 11}},
            "yaxis": {"title": {"text": ytitle, "standoff": 10}, "gridcolor": "#2d3148", "zerolinecolor": "#2d3148", "tickfont": {"size": 11}},
            "legend": {"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center", "font": {"size": 11}, "bgcolor": "transparent"},
            "hovermode": "x unified",
        }
        if tickfmt: l["yaxis"]["tickformat"] = tickfmt
        if yrange: l["yaxis"]["range"] = yrange
        return l

    def trace(name, x, y, idx, mode="lines+markers", ms=4, dash=None, fill=None, fc=None):
        t = {"name": name, "x": x, "y": y, "type": "scatter", "mode": mode,
             "line": {"color": PALETTE[idx % len(PALETTE)], "width": 2.2},
             "marker": {"size": ms, "color": PALETTE[idx % len(PALETTE)]},
             "hovertemplate": "%{y:.4f}<extra>" + name + "</extra>"}
        if dash: t["line"]["dash"] = dash
        if fill: t["fill"] = fill
        if fc: t["fillcolor"] = fc
        return t

    # Accuracy
    max_acc = 0
    acc_traces = []
    for i, lab in enumerate(labels):
        d = experiments[lab]
        if d["acc_values"]:
            acc_traces.append(trace(lab, d["acc_rounds"], d["acc_values"], i, ms=5))
            max_acc = max(max_acc, max(d["acc_values"]))
    if acc_traces:
        charts["accuracy"] = {"traces": acc_traces, "layout": layout("Accuracy", ".0%", [0, max_acc * 1.08])}

    # Common metrics
    info = classify_metrics(experiments)

    def pick_x(exp, metric_name, vals):
        """Pick the best x-axis for a metric based on length and source."""
        dr = exp["drift_rounds"]
        if len(dr) == len(vals):
            return dr
        # For g_measurement metrics, use their actual round numbers
        gmr = exp.get("g_meas_rounds", [])
        if metric_name.startswith("g_") and len(gmr) == len(vals):
            return gmr
        return list(range(1, len(vals) + 1))

    for m in info["common"]:
        traces_list = []
        for i, lab in enumerate(labels):
            vals = experiments[lab]["metrics"].get(m)
            if vals:
                x = pick_x(experiments[lab], m, vals)
                traces_list.append(trace(lab, x, vals, i))
        if traces_list:
            charts[f"common_{m}"] = {"traces": traces_list, "layout": layout(m)}

    # Per-experiment unique metrics
    for lab, unique_metrics in info["per_experiment"].items():
        exp = experiments[lab]
        idx = labels.index(lab)
        for m in sorted(unique_metrics):
            vals = exp["metrics"].get(m)
            if vals:
                x = pick_x(exp, m, vals)
                charts[f"unique_{lab}_{m}"] = {
                    "traces": [trace(m, x, vals, idx, ms=3)],
                    "layout": layout(m)
                }

    # V-value charts
    for i, lab in enumerate(labels):
        d = experiments[lab]
        if d["v_values"]:
            vr = list(range(1, len(d["v_values"]) + 1))
            charts[f"vvalue_{lab}"] = {
                "traces": [trace("V-Value", vr, d["v_values"], i, mode="lines", fill="tozeroy",
                                 fc=f"rgba({','.join(str(int(PALETTE[i%len(PALETTE)][j:j+2],16)) for j in (1,3,5))},0.08)")],
                "layout": layout("V-Value", xtitle="Epoch")
            }

    # Time record charts
    for i, lab in enumerate(labels):
        d = experiments[lab]
        if d["time_record"]:
            tr = list(range(1, len(d["time_record"]) + 1))
            charts[f"time_{lab}"] = {
                "traces": [trace("Elapsed (sec)", tr, d["time_record"], i, mode="lines")],
                "layout": layout("Elapsed Time (sec)", xtitle="Epoch")
            }

    return charts


# ─────────────────────────────────────────────
# 3. HTML 생성
# ─────────────────────────────────────────────

def build_config_table(experiments: dict) -> list:
    """설정 비교 테이블 데이터 생성"""
    labels = list(experiments.keys())
    keys = [
        ("Method", lambda c: c.get("method", "N/A")),
        ("Dataset", lambda c: c.get("dataset", "N/A")),
        ("Model", lambda c: c.get("model", "N/A")),
        ("Distribution", lambda c: c.get("distributer", c.get("distribution", "N/A"))),
        ("Dirichlet α", lambda c: str(c.get("dirichlet_alpha", c.get("alpha", "N/A")))),
        ("Selector", lambda c: c.get("selector", "N/A")),
        ("Batch Size", lambda c: str(c.get("batch_size", "N/A"))),
        ("Learning Rate", lambda c: str(c.get("learning_rate", c.get("lr", "N/A")))),
        ("Rounds/Epochs", lambda c: str(c.get("global_round", c.get("epochs", "N/A")))),
        ("Clients", lambda c: str(c.get("num_clients", c.get("users", "N/A")))),
        ("Clients/Round", lambda c: str(c.get("num_clients_per_round", c.get("participating", "N/A")))),
        ("Split Layer", lambda c: str(c.get("split_layer", "N/A"))),
        ("Grad Shuffle", lambda c: str(c.get("gradient_shuffle", c.get("generate", "N/A")))),
        ("Aggregator", lambda c: str(c.get("aggregator", "N/A"))),
        ("Optimizer", lambda c: str(c.get("optimizer", "N/A"))),
        ("Local Epochs", lambda c: str(c.get("local_epochs", "N/A"))),
        ("Momentum", lambda c: str(c.get("momentum", "N/A"))),
    ]
    rows = []
    for name, ext in keys:
        row = [name]
        for lab in labels:
            row.append(ext(experiments[lab]["config"]))
        # Skip rows where all experiments have N/A
        if any(cell != "N/A" and cell != "None" for cell in row[1:]):
            rows.append(row)
    return rows


def get_plotly_js() -> str:
    """Plotly.js를 찾아 반환 (pip 패키지 → CDN fallback)"""
    try:
        import plotly as _plotly
        js_path = os.path.join(os.path.dirname(_plotly.__file__), "package_data", "plotly.min.js")
        if os.path.exists(js_path):
            with open(js_path) as f:
                print(f"  📦 Using bundled plotly.min.js ({os.path.getsize(js_path)/1024/1024:.1f} MB)")
                return f.read()
    except ImportError:
        pass

    # Fallback to CDN (requires internet)
    print("  ⚠ plotly package not found, using CDN (requires internet)")
    return None


def generate_html(experiments: dict, output_path: str):
    """최종 HTML 대시보드 생성"""
    labels = list(experiments.keys())
    n_exp = len(labels)
    charts = build_chart_traces(experiments)
    config_rows = build_config_table(experiments)
    info = classify_metrics(experiments)

    plotly_js = get_plotly_js()
    if plotly_js:
        plotly_tag = f"<script>{plotly_js}</script>"
    else:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    # Count total unique metrics
    all_metric_names = set()
    for exp in experiments.values():
        all_metric_names.update(exp["metrics"].keys())
    has_vvalue = any(exp["v_values"] for exp in experiments.values())
    has_time = any(exp["time_record"] for exp in experiments.values())
    total_metrics = len(all_metric_names) + (1 if has_vvalue else 0) + (1 if has_time else 0) + 1  # +1 for accuracy

    # Summary data
    summary = []
    for lab in labels:
        d = experiments[lab]
        if d["acc_values"]:
            mx = max(d["acc_values"])
            mi = d["acc_values"].index(mx)
            summary.append({"label": lab, "max_acc": mx, "max_round": d["acc_rounds"][mi],
                            "last_acc": d["acc_values"][-1], "n_evals": len(d["acc_values"])})
        else:
            summary.append({"label": lab, "max_acc": 0, "max_round": 0, "last_acc": 0, "n_evals": 0})

    # Build tab structure dynamically
    tabs = []

    # Overview
    overview_charts = []
    if "accuracy" in charts:
        overview_charts.append(("accuracy", "Test Accuracy over Rounds", "Global model accuracy at each evaluation point", True))
    top_common = [m for m in ["G_drift", "M_norm", "A_cos", "G_drift_norm"] if f"common_{m}" in charts][:2]
    for m in top_common:
        overview_charts.append((f"common_{m}", m, "All experiments compared", False))
    tabs.append(("overview", "Overview", overview_charts))

    # Common metrics tab
    common_charts = []
    for m in info["common"]:
        key = f"common_{m}"
        if key in charts:
            common_charts.append((key, m, f"Shared across all {n_exp} experiments", False))
    if common_charts:
        tabs.append(("common", f"Common Metrics ({len(common_charts)})", common_charts))

    # Per-experiment unique metrics
    for lab in labels:
        unique = info["per_experiment"].get(lab, set())
        if not unique and not experiments[lab]["v_values"] and not experiments[lab]["time_record"]:
            continue
        exp_charts = []
        for m in sorted(unique):
            key = f"unique_{lab}_{m}"
            if key in charts:
                exp_charts.append((key, m, f"{lab} only", False))
        if experiments[lab]["v_values"]:
            key = f"vvalue_{lab}"
            if key in charts:
                exp_charts.append((key, "V-Value", "Generation quality indicator", False))
        if experiments[lab]["time_record"]:
            key = f"time_{lab}"
            if key in charts:
                exp_charts.append((key, "Training Time", "Cumulative elapsed time", False))
        if exp_charts:
            short_label = lab[:25] + ("…" if len(lab) > 25 else "")
            tabs.append((f"exp_{labels.index(lab)}", f"{short_label} Only ({len(exp_charts)})", exp_charts))

    tabs.append(("config", "Config Table", []))

    # Generate color CSS for up to N experiments
    color_css = "\n".join(
        f".summary-card:nth-child({i+1})::before{{background:{PALETTE[i%len(PALETTE)]}}}"
        f".summary-card:nth-child({i+1}) .value{{color:{PALETTE[i%len(PALETTE)]}}}"
        f".config-table td:nth-child({i+2}){{color:{PALETTE[i%len(PALETTE)]}}}"
        for i in range(n_exp)
    )

    # Build chart divs HTML and chartMap JS
    chart_div_html = {}
    chart_map_entries = []
    div_counter = 0
    for tab_id, tab_name, tab_charts in tabs:
        divs = []
        for chart_key, title, desc, is_full in tab_charts:
            div_id = f"c{div_counter}"
            div_counter += 1
            divs.append((div_id, title, desc, is_full))
            chart_map_entries.append(f"'{div_id}':'{chart_key}'")
        chart_div_html[tab_id] = divs

    # Build HTML sections
    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" data-tab="{tid}">{tname}</button>'
        for i, (tid, tname, _) in enumerate(tabs)
    )

    sections_html = ""
    for i, (tid, tname, _) in enumerate(tabs):
        active = " active" if i == 0 else ""
        if tid == "config":
            sections_html += f'<div class="chart-section{active}" id="tab-{tid}"><div class="chart-card"><h3>Experiment Configuration Comparison</h3><table class="config-table" id="configTable"></table></div></div>\n'
            continue

        divs = chart_div_html.get(tid, [])
        inner = ""
        pair_buf = []
        for div_id, title, desc, is_full in divs:
            card = f'<div class="chart-card"><h3>{title}</h3><p class="desc">{desc}</p><div id="{div_id}" style="height:360px"></div></div>'
            if is_full:
                if pair_buf:
                    inner += f'<div class="chart-row">{"".join(pair_buf)}</div>\n'
                    pair_buf = []
                inner += f'<div class="chart-row full">{card}</div>\n'
            else:
                pair_buf.append(card)
                if len(pair_buf) == 2:
                    inner += f'<div class="chart-row">{"".join(pair_buf)}</div>\n'
                    pair_buf = []
        if pair_buf:
            inner += f'<div class="chart-row">{"".join(pair_buf)}</div>\n'

        sections_html += f'<div class="chart-section{active}" id="tab-{tid}">{inner}</div>\n'

    # Initial render (first tab)
    first_tab_divs = chart_div_html.get(tabs[0][0], [])
    initial_render = ",".join(f"'{d[0]}'" for d in first_tab_divs)

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFL Experiment Dashboard</title>
{plotly_tag}
<style>
:root{{--bg:#0f1117;--surface:#1a1d29;--border:#2d3148;--text:#e4e6f0;--text-dim:#8b8fa3}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
.header{{background:linear-gradient(135deg,#1a1d29 0%,#252940 100%);border-bottom:1px solid var(--border);padding:28px 40px}}
.header h1{{font-size:1.5rem;font-weight:700;letter-spacing:-.02em;margin-bottom:4px}}
.header p{{color:var(--text-dim);font-size:.88rem}}
.header .badge{{display:inline-block;background:rgba(108,158,255,.15);color:#6c9eff;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:600;margin-left:8px}}
.container{{max-width:1440px;margin:0 auto;padding:24px 40px}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;margin-bottom:28px}}
.summary-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;position:relative;overflow:hidden}}
.summary-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
.summary-card .lbl{{font-size:.75rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
.summary-card .method{{font-size:.82rem;font-weight:600;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.summary-card .value{{font-size:1.8rem;font-weight:700}}
.summary-card .sub{{font-size:.75rem;color:var(--text-dim);margin-top:2px}}
{color_css}
.tab-nav{{display:flex;gap:4px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:4px;margin-bottom:20px;flex-wrap:wrap}}
.tab-btn{{padding:10px 18px;border:none;background:transparent;color:var(--text-dim);font-size:.82rem;font-weight:500;border-radius:8px;cursor:pointer;transition:all .2s;white-space:nowrap}}
.tab-btn:hover{{background:rgba(108,158,255,.1);color:var(--text)}}
.tab-btn.active{{background:rgba(108,158,255,.15);color:#6c9eff;font-weight:600}}
.chart-section{{display:none}}.chart-section.active{{display:block}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.chart-row.full{{grid-template-columns:1fr}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px}}
.chart-card h3{{font-size:.9rem;font-weight:600;margin-bottom:4px}}
.chart-card .desc{{font-size:.78rem;color:var(--text-dim);margin-bottom:10px}}
.config-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
.config-table th,.config-table td{{padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}}
.config-table th{{color:var(--text-dim);font-weight:600;text-transform:uppercase;font-size:.72rem;letter-spacing:.05em}}
.config-table tr:hover td{{background:rgba(108,158,255,.04)}}
.config-table td:first-child{{font-weight:600;color:var(--text);min-width:130px}}
@media(max-width:900px){{.summary-grid{{grid-template-columns:1fr}}.chart-row{{grid-template-columns:1fr}}.container{{padding:12px 8px}}.header{{padding:18px 16px}}.chart-card{{padding:14px 10px}}.chart-card h3{{font-size:.95rem}}.chart-card .desc{{font-size:.72rem;margin-bottom:6px}}.tab-nav{{gap:2px;padding:3px}}.tab-btn{{padding:8px 12px;font-size:.78rem}}}}
/* Fullscreen modal for mobile chart zoom */
.chart-modal-overlay{{display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,.92);z-index:9999;flex-direction:column;align-items:stretch;justify-content:center;padding:0}}
.chart-modal-overlay.open{{display:flex}}
.chart-modal-header{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;flex-shrink:0}}
.chart-modal-header h3{{color:#e4e6f0;font-size:.95rem;font-weight:600;margin:0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.chart-modal-close{{background:rgba(255,255,255,.12);border:none;color:#e4e6f0;font-size:1.4rem;width:40px;height:40px;border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-left:12px;-webkit-tap-highlight-color:transparent}}
.chart-modal-close:active{{background:rgba(255,255,255,.25)}}
#modalChart{{flex:1;min-height:0;width:100%}}
/* Tap hint on mobile */
@media(max-width:900px){{.chart-card{{cursor:pointer;-webkit-tap-highlight-color:transparent}}.chart-card::after{{content:'Tap to expand';position:absolute;top:8px;right:10px;font-size:.65rem;color:var(--text-dim);opacity:.6;pointer-events:none}}.chart-card{{position:relative}}}}
</style>
</head>
<body>
<div class="header">
  <h1>SFL Experiment Dashboard <span class="badge">{n_exp} Experiments</span> <span class="badge">{total_metrics} Metrics</span></h1>
  <p>{os.path.abspath(os.path.dirname(output_path))} · Auto-generated from {n_exp} JSON files</p>
</div>
<div class="container">
  <div class="summary-grid" id="summaryGrid"></div>
  <div class="tab-nav">{tab_buttons}</div>
  {sections_html}
</div>
<!-- Fullscreen chart modal -->
<div class="chart-modal-overlay" id="chartModal">
  <div class="chart-modal-header">
    <h3 id="modalTitle"></h3>
    <button class="chart-modal-close" id="modalClose" aria-label="Close">✕</button>
  </div>
  <div id="modalChart"></div>
</div>
<script>
var CHARTS={json.dumps(charts)};
var SUMMARY={json.dumps(summary)};
var CONFIG_ROWS={json.dumps(config_rows)};
var LABELS={json.dumps(labels)};
var pcfg={{responsive:true,displayModeBar:true,modeBarButtonsToRemove:['lasso2d','select2d']}};

// Summary cards
var sg=document.getElementById('summaryGrid');
SUMMARY.forEach(function(s,i){{
  sg.innerHTML+='<div class="summary-card"><div class="lbl">Exp '+(i+1)+'</div><div class="method">'+s.label+'</div><div class="value">'+(s.max_acc*100).toFixed(1)+'%</div><div class="sub">Best @ Round '+s.max_round+' · Last: '+(s.last_acc*100).toFixed(1)+'% · '+s.n_evals+' evals</div></div>';
}});

// Chart map
var chartMap={{{",".join(chart_map_entries)}}};
var rendered={{}};
function renderChart(id){{
  if(rendered[id])return;
  var key=chartMap[id];
  if(key&&CHARTS[key]){{Plotly.newPlot(id,CHARTS[key].traces,CHARTS[key].layout,pcfg);rendered[id]=true;}}
}}

// Initial render
[{initial_render}].forEach(renderChart);

// Config table
var ct=document.getElementById('configTable');
if(ct){{
  var h='<thead><tr><th>Parameter</th>';
  LABELS.forEach(function(l){{h+='<th>'+l+'</th>';}});
  h+='</tr></thead><tbody>';
  CONFIG_ROWS.forEach(function(r){{h+='<tr>';r.forEach(function(c){{h+='<td>'+c+'</td>';}});h+='</tr>';}});
  ct.innerHTML=h+'</tbody>';
}}

// Tab switching
document.querySelectorAll('.tab-btn').forEach(function(btn){{
  btn.addEventListener('click',function(){{
    document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});
    document.querySelectorAll('.chart-section').forEach(function(s){{s.classList.remove('active');}});
    btn.classList.add('active');
    var tab=document.getElementById('tab-'+btn.dataset.tab);
    tab.classList.add('active');
    tab.querySelectorAll('[id^="c"]').forEach(function(el){{if(chartMap[el.id])renderChart(el.id);}});
    window.dispatchEvent(new Event('resize'));
  }});
}});

// ─── Fullscreen modal for chart zoom (mobile-friendly) ───
var overlay=document.getElementById('chartModal');
var modalChart=document.getElementById('modalChart');
var modalTitle=document.getElementById('modalTitle');
var currentModalKey=null;

function openModal(divId){{
  var key=chartMap[divId];
  if(!key||!CHARTS[key])return;
  currentModalKey=key;
  var ch=CHARTS[key];
  modalTitle.textContent=ch.layout.yaxis&&ch.layout.yaxis.title?ch.layout.yaxis.title.text||ch.layout.yaxis.title:'';
  overlay.classList.add('open');
  document.body.style.overflow='hidden';
  // Render with mobile-optimized layout
  var ml=JSON.parse(JSON.stringify(ch.layout));
  ml.margin={{t:10,r:12,b:50,l:50}};
  ml.legend={{orientation:'h',y:-0.12,x:0.5,xanchor:'center',font:{{size:12}},bgcolor:'transparent'}};
  if(ml.font)ml.font.size=13;
  Plotly.newPlot(modalChart,ch.traces,ml,{{responsive:true,displayModeBar:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','toImage']}});
}}

function closeModal(){{
  overlay.classList.remove('open');
  document.body.style.overflow='';
  Plotly.purge(modalChart);
  currentModalKey=null;
}}

document.getElementById('modalClose').addEventListener('click',closeModal);
overlay.addEventListener('click',function(e){{if(e.target===overlay)closeModal();}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape'&&currentModalKey)closeModal();}});

// Attach tap-to-zoom on all chart cards
document.querySelectorAll('.chart-card').forEach(function(card){{
  var chartDiv=card.querySelector('[id^="c"]');
  if(chartDiv&&chartMap[chartDiv.id]){{
    card.addEventListener('click',function(e){{
      // Don't intercept Plotly toolbar clicks
      if(e.target.closest('.modebar'))return;
      openModal(chartDiv.id);
    }});
  }}
}});
</script>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n🎉 Dashboard saved: {output_path} ({size_mb:.1f} MB)")
    print(f"   {n_exp} experiments · {total_metrics} metrics · {len(charts)} charts")
    return output_path


# ─────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split Federated Learning 실험 결과 대시보드 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sfl_dashboard.py ./results/
  python sfl_dashboard.py ./results/ --output my_report.html
  python sfl_dashboard.py ./results/ --open
        """,
    )
    parser.add_argument("directory", help="JSON 결과 파일들이 있는 디렉토리 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 HTML 파일 경로 (기본: <directory>/sfl_dashboard.html)")
    parser.add_argument("--open", action="store_true", help="생성 후 브라우저에서 자동 열기")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ Directory not found: {args.directory}")
        sys.exit(1)

    output = args.output or os.path.join(args.directory, "sfl_dashboard.html")

    print(f"\n{'='*50}")
    print(f"  SFL Experiment Dashboard Generator")
    print(f"{'='*50}\n")

    experiments = load_all_experiments(args.directory)
    if not experiments:
        print("❌ No valid experiment files found")
        sys.exit(1)

    path = generate_html(experiments, output)

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(path)}")


if __name__ == "__main__":
    main()
