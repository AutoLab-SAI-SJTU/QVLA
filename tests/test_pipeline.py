import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qvla_pipeline", ROOT / "scripts/run_qvla.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def environment():
    env = dict(os.environ)
    for key in (
        "QVLA_ATTENTION",
        "QVLA_CALIBRATION_ATTENTION",
        "QVLA_RESTORE_PRUNED_FRACTION",
        "QVLA_BACKEND_ROOT",
    ):
        env.pop(key, None)
    return env


@pytest.mark.parametrize(
    "backend,bits,attention,repair",
    [
        ("openvla", 4, "sdpa", 0.0),
        ("openvla", 8, "flash_attention_2", 0.25),
        ("openvla-oft", 4, "sdpa", 0.0),
        ("openvla-oft", 8, "sdpa", 0.0),
    ],
)
def test_plan_defaults_for_each_weight_only_route(
    tmp_path, backend, bits, attention, repair
):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_qvla.py"),
            "--backend",
            backend,
            "--action",
            "plan",
            "--target-avg-bits",
            str(bits),
            "--output-root",
            str(tmp_path / "run"),
        ],
        env=environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["bits"] == "0,4,8,16" and plan["samples"] == 512
    assert plan["target_avg_bits"] == bits and plan["attention"] == attention
    assert plan["seed"] is None and plan["seed_policy"] == "random_per_run"
    assert (
        plan["calibration_attention"] == "sdpa"
        and plan["restore_pruned_fraction"] == repair
    )
    assert not (tmp_path / "run").exists()


def test_oft_rejects_causal_attention_and_openvla_pruning_repair(tmp_path):
    for option in (("--attention", "eager"), ("--restore-pruned-fraction", "0.25")):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_qvla.py"),
                "--backend",
                "openvla-oft",
                "--action",
                "plan",
                "--output-root",
                str(tmp_path / "run"),
                *option,
            ],
            env=environment(),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "OFT uses bidirectional SDPA" in result.stderr


def test_verified_cache_rejects_changed_inputs_and_corrupted_artifacts(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(output_root=tmp_path, backend_root=tmp_path)
    artifact = tmp_path / "artifact.txt"
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        artifact.write_text("verified")

    monkeypatch.setattr(pipeline.subprocess, "run", child)
    pipeline.stage(args, "test", {"checkpoint": "A"}, artifact, ["test"])
    pipeline.stage(args, "test", {"checkpoint": "A"}, artifact, ["test"])
    assert len(calls) == 1
    with pytest.raises(ValueError, match="stale"):
        pipeline.stage(args, "test", {"checkpoint": "B"}, artifact, ["test"])
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        pipeline.stage(args, "test", {"checkpoint": "A"}, artifact, ["test"])


def test_untracked_output_is_never_reused(tmp_path):
    artifact = tmp_path / "proxy.pt"
    artifact.write_bytes(b"old incomplete run")
    with pytest.raises(ValueError, match="Unverified"):
        pipeline.stage(
            SimpleNamespace(output_root=tmp_path), "proxy", {}, artifact, ["unused"]
        )


def test_proxy_filename_contract_fails_before_loading_a_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_qvla.py"),
            "--action",
            "proxy",
            "--checkpoint",
            str(tmp_path / "unused_checkpoint"),
            "--output-root",
            str(tmp_path / "run"),
            "--proxy_pt",
            str(tmp_path / "wrong.pt"),
        ],
        env=environment(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Calibration writes proxy.pt" in result.stderr
    assert not (tmp_path / "run").exists()


def test_random_seed_is_per_run_and_reused_across_stages(tmp_path, monkeypatch):
    seeds = iter([123, 456])
    monkeypatch.setattr(pipeline.secrets, "randbits", lambda bits: next(seeds))
    first, second = tmp_path / "first", tmp_path / "second"
    assert pipeline.resolve_run_seed(first) == 123
    assert pipeline.resolve_run_seed(first) == 123
    assert pipeline.resolve_run_seed(second) == 456
    assert json.loads((first / ".qvla_seed.json").read_text())["selection"] == "random"


def test_explicit_seed_replay_and_conflict_rejection(tmp_path):
    assert pipeline.resolve_run_seed(tmp_path, 19) == 19
    assert pipeline.resolve_run_seed(tmp_path) == 19
    with pytest.raises(ValueError, match="fresh output directory"):
        pipeline.resolve_run_seed(tmp_path, 20)
    assert pipeline.resolve_run_seed(tmp_path) == 19


def test_seed_planning_is_read_only(tmp_path):
    output = tmp_path / "run"
    assert pipeline.resolve_run_seed(output, persist=False) is None
    assert pipeline.resolve_run_seed(output, 19, persist=False) == 19
    assert not output.exists()
    pipeline.resolve_run_seed(output, 23)
    assert pipeline.resolve_run_seed(output, persist=False) == 23
