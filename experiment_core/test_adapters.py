"""
experiment_core 통합 검증 스크립트.

GPU 서버에서 실행:
    cd USFL-fork
    python -m experiment_core.test_adapters

테스트 항목:
  1. SFL Adapter - command 빌드 + 인자 매핑
  2. USFL batch_size 로직 (server_batch_size vs client_batch_size)
  3. GAS Adapter - 모델 매핑 + validation
  4. GAS Adapter - 분포 모드 환경변수 매핑
  5. GAS Adapter - command 빌드
  6. MultiSFL Adapter - command 빌드 + override
  7. Runner - subprocess output capture 존재 확인
  8. Normalization - 3개 프레임워크 정규화
  9. Edge cases (empty data, invalid framework, etc.)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

# project root = 이 스크립트의 부모의 부모
REPO_ROOT = Path(__file__).resolve().parent.parent

# experiment_core가 import 가능하도록
sys.path.insert(0, str(REPO_ROOT))

from experiment_core.spec import ExperimentSpec, _deep_merge, _defaults, _validate
from experiment_core.adapters import get_adapter
from experiment_core.normalization import normalize_sfl, normalize_gas, normalize_multisfl


passed = 0
failed = 0


def _test(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


def _make_spec(overrides: dict) -> dict:
    merged = _deep_merge(_defaults(), overrides)
    _validate(merged)
    return merged


# =========================================================
# TEST 1: SFL command build
# =========================================================
def test_sfl_command_build():
    print("\n" + "=" * 60)
    print("TEST 1: SFL Adapter - command build")
    print("=" * 60)

    spec = _make_spec({
        "framework": "sfl",
        "method": "sfl",
        "common": {
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "rounds": 100,
            "distribution": {"mode": "shard_dirichlet"},
        },
    })
    adapter = get_adapter("sfl")
    cmd = adapter.build_command(spec, REPO_ROOT)
    env = adapter.build_env(spec)
    cwd = adapter.default_cwd(REPO_ROOT)

    _test("CWD is sfl_framework dir", lambda: assert_(
        "sfl_framework" in str(cwd),
        f"CWD should contain sfl_framework, got {cwd}",
    ))
    _test("entry point is server/main.py", lambda: assert_(
        cmd[1] == "server/main.py",
        f"Expected server/main.py, got {cmd[1]}",
    ))

    for flag in [
        "--dataset", "--model", "--method", "--global-round",
        "--split-layer", "--distributer", "--num-clients",
        "--num-clients-per-round", "--enable-drift-measurement",
    ]:
        _test(f"has {flag}", lambda f=flag: assert_(f in cmd, f"Missing {f}"))

    _test("dataset=cifar10", lambda: assert_("cifar10" in cmd))
    _test("rounds=100", lambda: assert_("100" in cmd))


# =========================================================
# TEST 2: USFL batch_size logic
# =========================================================
def test_usfl_batch_size():
    print("\n" + "=" * 60)
    print("TEST 2: USFL batch_size logic")
    print("=" * 60)

    adapter = get_adapter("sfl")

    # USFL -> server_batch_size
    usfl_spec = _make_spec({
        "framework": "sfl",
        "method": "usfl",
        "common": {"client_batch_size": 50, "server_batch_size": 500},
    })
    cmd = adapter.build_command(usfl_spec, REPO_ROOT)
    bs_idx = cmd.index("--batch-size") + 1
    _test("USFL uses server_batch_size=500", lambda: assert_(
        cmd[bs_idx] == "500", f"Got {cmd[bs_idx]}",
    ))

    # SFL -> client_batch_size
    sfl_spec = _make_spec({
        "framework": "sfl",
        "method": "sfl",
        "common": {"client_batch_size": 50, "server_batch_size": 500},
    })
    cmd2 = adapter.build_command(sfl_spec, REPO_ROOT)
    bs_idx2 = cmd2.index("--batch-size") + 1
    _test("SFL uses client_batch_size=50", lambda: assert_(
        cmd2[bs_idx2] == "50", f"Got {cmd2[bs_idx2]}",
    ))


# =========================================================
# TEST 3: GAS model validation
# =========================================================
def test_gas_model_validation():
    print("\n" + "=" * 60)
    print("TEST 3: GAS Adapter - model validation")
    print("=" * 60)

    adapter = get_adapter("gas")

    # resnet18_flex -> resnet18
    spec1 = _make_spec({"framework": "gas", "common": {"model": "resnet18_flex"}})
    env1 = adapter.build_env(spec1)
    _test("resnet18_flex -> resnet18", lambda: assert_(
        env1["GAS_MODEL"] == "resnet18", f"Got {env1['GAS_MODEL']}",
    ))

    # alexnet -> alexnet
    spec2 = _make_spec({"framework": "gas", "common": {"model": "alexnet"}})
    env2 = adapter.build_env(spec2)
    _test("alexnet -> alexnet", lambda: assert_(
        env2["GAS_MODEL"] == "alexnet", f"Got {env2['GAS_MODEL']}",
    ))

    # vgg16 -> ValueError
    def _vgg_should_fail():
        spec_bad = _make_spec({"framework": "gas", "common": {"model": "vgg16"}})
        adapter.build_env(spec_bad)

    _test("vgg16 raises ValueError", lambda: assert_raises(ValueError, _vgg_should_fail))

    # vgg11 -> ValueError
    def _vgg11_should_fail():
        spec_bad = _make_spec({"framework": "gas", "common": {"model": "vgg11"}})
        adapter.build_env(spec_bad)

    _test("vgg11 raises ValueError", lambda: assert_raises(ValueError, _vgg11_should_fail))


# =========================================================
# TEST 4: GAS distribution mode mapping
# =========================================================
def test_gas_distribution_mode():
    print("\n" + "=" * 60)
    print("TEST 4: GAS Adapter - distribution mode env vars")
    print("=" * 60)

    adapter = get_adapter("gas")

    cases = [
        ("shard_dirichlet", {"GAS_IID": "false", "GAS_DIRICHLET": "false", "GAS_LABEL_DIRICHLET": "true"}),
        ("iid", {"GAS_IID": "true", "GAS_DIRICHLET": "false", "GAS_LABEL_DIRICHLET": "false"}),
        ("dirichlet", {"GAS_IID": "false", "GAS_DIRICHLET": "true", "GAS_LABEL_DIRICHLET": "false"}),
    ]

    for mode, expected_flags in cases:
        spec = _make_spec({
            "framework": "gas",
            "common": {"model": "resnet18", "distribution": {"mode": mode}},
        })
        env = adapter.build_env(spec)
        for key, expected_val in expected_flags.items():
            _test(f"mode={mode}: {key}={expected_val}", lambda k=key, v=expected_val: assert_(
                env.get(k) == v, f"Got {env.get(k)}",
            ))


# =========================================================
# TEST 5: GAS command build
# =========================================================
def test_gas_command():
    print("\n" + "=" * 60)
    print("TEST 5: GAS Adapter - command build")
    print("=" * 60)

    adapter = get_adapter("gas")
    spec = _make_spec({"framework": "gas", "common": {"model": "resnet18"}})
    cmd = adapter.build_command(spec, REPO_ROOT)

    _test("entry point is GAS_main.py", lambda: assert_(
        "GAS_main.py" in cmd[1], f"Got {cmd[1]}",
    ))
    _test("3rd arg is use_variance_g bool", lambda: assert_(
        cmd[2] in ("true", "false"), f"Got {cmd[2]}",
    ))

    # with variance_g override
    spec_vg = _make_spec({
        "framework": "gas",
        "common": {"model": "resnet18"},
        "framework_overrides": {"use_variance_g": True},
    })
    cmd_vg = adapter.build_command(spec_vg, REPO_ROOT)
    _test("use_variance_g=True -> 'true'", lambda: assert_(
        cmd_vg[2] == "true", f"Got {cmd_vg[2]}",
    ))


# =========================================================
# TEST 6: MultiSFL command build
# =========================================================
def test_multisfl_command():
    print("\n" + "=" * 60)
    print("TEST 6: MultiSFL Adapter - command build")
    print("=" * 60)

    adapter = get_adapter("multisfl")
    spec = _make_spec({
        "framework": "multisfl",
        "common": {
            "dataset": "cifar10",
            "model": "resnet18_flex",
            "rounds": 50,
            "clients_per_round": 10,
            "distribution": {"mode": "shard_dirichlet"},
        },
        "framework_overrides": {"branches": 3, "lr_server": 0.01},
    })
    cmd = adapter.build_command(spec, REPO_ROOT)
    cwd = adapter.default_cwd(REPO_ROOT)

    _test("CWD is multisfl_implementation", lambda: assert_(
        "multisfl_implementation" in str(cwd),
    ))
    _test("entry point is run_multisfl.py", lambda: assert_(
        cmd[1] == "run_multisfl.py",
    ))

    for flag in [
        "--dataset", "--model_type", "--partition", "--n_main",
        "--lr_client", "--lr_server", "--branches",
        "--enable_drift_measurement",
    ]:
        _test(f"has {flag}", lambda f=flag: assert_(f in cmd, f"Missing {f}"))

    lr_idx = cmd.index("--lr_server") + 1
    _test("lr_server=0.01", lambda: assert_(cmd[lr_idx] == "0.01", f"Got {cmd[lr_idx]}"))

    br_idx = cmd.index("--branches") + 1
    _test("branches=3", lambda: assert_(cmd[br_idx] == "3", f"Got {cmd[br_idx]}"))


# =========================================================
# TEST 7: Runner subprocess capture
# =========================================================
def test_runner_capture():
    print("\n" + "=" * 60)
    print("TEST 7: Runner - subprocess output capture")
    print("=" * 60)

    from experiment_core import runner
    src = inspect.getsource(runner.run_experiment)

    _test("has capture_output=True", lambda: assert_("capture_output=True" in src))
    _test("has text=True", lambda: assert_("text=True" in src))
    _test("handles proc.stdout", lambda: assert_("proc.stdout" in src))
    _test("handles proc.stderr", lambda: assert_("proc.stderr" in src))


# =========================================================
# TEST 8: Normalization - all 3 frameworks
# =========================================================
def test_normalization():
    print("\n" + "=" * 60)
    print("TEST 8: Normalization - all 3 frameworks")
    print("=" * 60)

    # SFL
    sfl_raw = {
        "config": {"method": "sfl"},
        "metric": {
            "1": [{"event": "MODEL_EVALUATED", "params": {"accuracy": 0.75}}],
            "2": [{"event": "MODEL_EVALUATED", "params": {"accuracy": 0.80}}],
        },
        "drift_history": {"A_cos": [0.5, 0.6], "M_norm": [0.1, 0.2], "n_valid_alignment": [8, 9]},
    }
    sfl_norm = normalize_sfl(sfl_raw, Path("/tmp/t.json"), {"framework": "sfl"})
    _test("SFL accuracy=[0.75, 0.80]", lambda: assert_(
        sfl_norm["accuracy_by_round"] == [0.75, 0.80],
        f"Got {sfl_norm['accuracy_by_round']}",
    ))
    _test("SFL alignment A_cos", lambda: assert_(
        sfl_norm["alignment_history"]["A_cos"] == [0.5, 0.6],
    ))
    _test("SFL schema_version=1.0", lambda: assert_(
        sfl_norm["schema_version"] == "1.0",
    ))

    # GAS
    gas_raw = {
        "config": {"method": "gas"},
        "accuracy": [0.3, 0.5, 0.7],
        "drift_history": {"A_cos": [0.4], "M_norm": [0.2]},
        "g_history": {"scores": [1, 2]},
    }
    gas_norm = normalize_gas(gas_raw, Path("/tmp/t.json"), {"framework": "gas"})
    _test("GAS accuracy=[0.3, 0.5, 0.7]", lambda: assert_(
        gas_norm["accuracy_by_round"] == [0.3, 0.5, 0.7],
    ))
    _test("GAS g_history preserved", lambda: assert_(
        gas_norm["g_history"]["scores"] == [1, 2],
    ))

    # MultiSFL
    msfl_raw = {
        "config": {"method": "multisfl"},
        "rounds": [{"accuracy": 0.4}, {"accuracy": 0.6}],
        "drift_history": {
            "A_cos_client": [0.3], "M_norm_client": [0.1],
            "A_cos_server": [0.5], "M_norm_server": [0.2],
            "n_valid_alignment": [10],
        },
        "summary": {"final_acc": 0.6},
    }
    msfl_norm = normalize_multisfl(msfl_raw, Path("/tmp/t.json"), {"framework": "multisfl"})
    _test("MultiSFL accuracy=[0.4, 0.6]", lambda: assert_(
        msfl_norm["accuracy_by_round"] == [0.4, 0.6],
    ))
    _test("MultiSFL A_cos_client", lambda: assert_(
        msfl_norm["alignment_history"]["A_cos_client"] == [0.3],
    ))
    _test("MultiSFL A_cos_server", lambda: assert_(
        msfl_norm["alignment_history"]["A_cos_server"] == [0.5],
    ))
    _test("MultiSFL summary preserved", lambda: assert_(
        msfl_norm["summary"]["final_acc"] == 0.6,
    ))


# =========================================================
# TEST 9: Edge cases
# =========================================================
def test_edge_cases():
    print("\n" + "=" * 60)
    print("TEST 9: Edge cases")
    print("=" * 60)

    # Empty drift history
    norm = normalize_sfl({"config": {}, "metric": {}}, Path("/tmp/x.json"), {})
    _test("empty drift -> {}", lambda: assert_(norm["drift_history"] == {}))
    _test("empty alignment -> empty lists", lambda: assert_(
        norm["alignment_history"]["A_cos"] == [],
    ))

    # Missing accuracy
    norm2 = normalize_gas({"config": {}}, Path("/tmp/x.json"), {})
    _test("missing accuracy -> []", lambda: assert_(norm2["accuracy_by_round"] == []))

    # None accuracy in MultiSFL
    norm3 = normalize_multisfl(
        {"config": {}, "rounds": [{"no_acc": True}, {"accuracy": 0.5}]},
        Path("/tmp/x.json"), {},
    )
    _test("partial accuracy -> [None, 0.5]", lambda: assert_(
        norm3["accuracy_by_round"] == [None, 0.5],
        f"Got {norm3['accuracy_by_round']}",
    ))

    # Invalid framework
    _test("invalid framework -> ValueError", lambda: assert_raises(
        ValueError, lambda: get_adapter("invalid"),
    ))

    # Spec validation: missing framework
    _test("bad framework -> ValueError", lambda: assert_raises(
        ValueError, lambda: _validate({"framework": "bad_framework", "common": {}}),
    ))

    # Spec loading: nonexistent file
    from experiment_core.spec import load_spec
    _test("nonexistent spec -> FileNotFoundError", lambda: assert_raises(
        FileNotFoundError, lambda: load_spec("/nonexistent/path.json"),
    ))


# =========================================================
# Helpers
# =========================================================
def assert_(condition, msg=""):
    if not condition:
        raise AssertionError(msg or "Assertion failed")


def assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised")


# =========================================================
# Main
# =========================================================
def main():
    print("experiment_core Adapter Verification")
    print("=" * 60)
    print(f"REPO_ROOT: {REPO_ROOT}")

    test_sfl_command_build()
    test_usfl_batch_size()
    test_gas_model_validation()
    test_gas_distribution_mode()
    test_gas_command()
    test_multisfl_command()
    test_runner_capture()
    test_normalization()
    test_edge_cases()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
