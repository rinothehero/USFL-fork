from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _parse_sfl_accuracy(metric_blob: Any) -> List[Optional[float]]:
    if not isinstance(metric_blob, dict):
        return []

    rounds: List[int] = []
    for k in metric_blob.keys():
        try:
            rounds.append(int(k))
        except (TypeError, ValueError):
            continue
    rounds.sort()

    out: List[Optional[float]] = []
    for r in rounds:
        events = metric_blob.get(r) or metric_blob.get(str(r))
        acc: Optional[float] = None
        if isinstance(events, list):
            for e in events:
                if not isinstance(e, dict):
                    continue
                if e.get("event") == "MODEL_EVALUATED":
                    params = e.get("params", {})
                    if isinstance(params, dict) and "accuracy" in params:
                        try:
                            acc = float(params["accuracy"])
                        except (TypeError, ValueError):
                            acc = None
        out.append(acc)
    return out


def _extract_experiment_a_history(drift_history: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {
        "A_c_ratio": _safe_list(drift_history.get("expA_A_c_ratio")),
        "A_c_rel": _safe_list(drift_history.get("expA_A_c_rel")),
        "B_c": _safe_list(drift_history.get("expA_B_c")),
        "C_c": _safe_list(drift_history.get("expA_C_c")),
        "C_c_per_client_probe": _safe_list(
            drift_history.get("expA_C_c_per_client_probe")
        ),
        "B_s": _safe_list(drift_history.get("expA_B_s")),
        "m2_c": _safe_list(drift_history.get("expA_m2_c")),
        "u2_c": _safe_list(drift_history.get("expA_u2_c")),
        "var_c": _safe_list(drift_history.get("expA_var_c")),
        "server_mag_per_step": _safe_list(
            drift_history.get("expA_server_mag_per_step")
        ),
        "server_mag_per_step_sq": _safe_list(
            drift_history.get("expA_server_mag_per_step_sq")
        ),
        "per_round": _safe_list(drift_history.get("experiment_a")),
    }


def normalize_sfl(raw: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
    drift_history = raw.get("drift_history")
    if not isinstance(drift_history, dict):
        drift_history = {}

    alignment = {
        "A_cos": _safe_list(drift_history.get("A_cos")),
        "M_norm": _safe_list(drift_history.get("M_norm")),
        "n_valid_alignment": _safe_list(drift_history.get("n_valid_alignment")),
    }

    accuracy_by_round = _parse_sfl_accuracy(raw.get("metric"))

    return {
        "schema_version": "1.0",
        "normalized_at": _utc_now_iso(),
        "framework": "sfl",
        "run_meta": run_meta,
        "raw_result_path": str(raw_path),
        "config": raw.get("config", {}),
        "accuracy_by_round": accuracy_by_round,
        "drift_history": drift_history,
        "alignment_history": alignment,
        "experiment_a_history": _extract_experiment_a_history(drift_history),
        "g_measurements": _safe_list(raw.get("g_measurements")),
    }


def normalize_gas(raw: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
    drift_history = raw.get("drift_history")
    if not isinstance(drift_history, dict):
        drift_history = {}

    alignment = {
        "A_cos": _safe_list(drift_history.get("A_cos")),
        "M_norm": _safe_list(drift_history.get("M_norm")),
        "n_valid_alignment": _safe_list(drift_history.get("n_valid_alignment")),
    }

    return {
        "schema_version": "1.0",
        "normalized_at": _utc_now_iso(),
        "framework": "gas",
        "run_meta": run_meta,
        "raw_result_path": str(raw_path),
        "config": raw.get("config", {}),
        "accuracy_by_round": _safe_list(raw.get("accuracy")),
        "drift_history": drift_history,
        "alignment_history": alignment,
        "experiment_a_history": _extract_experiment_a_history(drift_history),
        "g_history": raw.get("g_history", {}),
    }


def normalize_multisfl(raw: Dict[str, Any], raw_path: Path, run_meta: Dict[str, Any]) -> Dict[str, Any]:
    drift_history = raw.get("drift_history")
    if not isinstance(drift_history, dict):
        drift_history = {}

    rounds = _safe_list(raw.get("rounds"))
    accuracy_by_round: List[Optional[float]] = []
    for row in rounds:
        if isinstance(row, dict) and "accuracy" in row:
            try:
                accuracy_by_round.append(float(row["accuracy"]))
            except (TypeError, ValueError):
                accuracy_by_round.append(None)
        else:
            accuracy_by_round.append(None)

    alignment = {
        "A_cos_client": _safe_list(drift_history.get("A_cos_client")),
        "M_norm_client": _safe_list(drift_history.get("M_norm_client")),
        "A_cos_server": _safe_list(drift_history.get("A_cos_server")),
        "M_norm_server": _safe_list(drift_history.get("M_norm_server")),
        "n_valid_alignment": _safe_list(drift_history.get("n_valid_alignment")),
    }

    return {
        "schema_version": "1.0",
        "normalized_at": _utc_now_iso(),
        "framework": "multisfl",
        "run_meta": run_meta,
        "raw_result_path": str(raw_path),
        "config": raw.get("config", {}),
        "accuracy_by_round": accuracy_by_round,
        "drift_history": drift_history,
        "alignment_history": alignment,
        "experiment_a_history": _extract_experiment_a_history(drift_history),
        "g_measurements": _safe_list(raw.get("g_measurements")),
        "summary": raw.get("summary", {}),
    }


def load_raw_result(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Raw result must be object JSON: {path}")
    return payload
