import json

from experiments.aggregate_real_runs import aggregate, collect_runs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_real_aggregator_uses_only_measured_runs(tmp_path):
    for seed, psnr in ((0, 30.0), (1, 32.0)):
        run_dir = tmp_path / f"measured_{seed}"
        write_json(run_dir / "run_manifest.json", {
            "evidence_type": "measured",
            "method": "ZeroGS",
            "scene": "bonsai",
            "hard_vram_limit_mb": 4096,
            "seed": seed,
        })
        write_json(run_dir / "run_status.json", {"status": "completed", "oom": False})
        write_json(run_dir / "test" / "ours_30000" / "results.json", {
            "PSNR": psnr,
            "SSIM": 0.9,
            "LPIPS": 0.1,
        })
        (run_dir / "training_telemetry.jsonl").write_text(
            json.dumps({"event": "training_end", "memory": {"peak_reserved_mb": 3500.0}}) + "\n",
            encoding="utf-8",
        )

    simulated = tmp_path / "simulated"
    write_json(simulated / "run_manifest.json", {
        "evidence_type": "simulation",
        "method": "ZeroGS",
        "scene": "bonsai",
        "hard_vram_limit_mb": 4096,
        "seed": 2,
    })
    write_json(simulated / "run_status.json", {"status": "completed", "oom": False})

    records = collect_runs(tmp_path)
    assert len(records) == 2
    summary = aggregate(records)
    assert len(summary) == 1
    assert summary[0]["psnr"]["mean"] == 31.0
    assert summary[0]["psnr"]["std"] > 0
    assert summary[0]["peak_reserved_mb"]["mean"] == 3500.0


def test_real_aggregator_counts_failed_oom_runs(tmp_path):
    run_dir = tmp_path / "failed"
    write_json(run_dir / "run_manifest.json", {
        "evidence_type": "measured",
        "method": "Vanilla",
        "scene": "garden",
        "hard_vram_limit_mb": 4096,
        "seed": 0,
    })
    write_json(run_dir / "run_status.json", {"status": "failed", "oom": True})
    summary = aggregate(collect_runs(tmp_path))
    assert summary[0]["oom_runs"] == 1
    assert summary[0]["completed_runs"] == 0
    assert summary[0]["psnr"]["mean"] is None


def test_real_aggregator_counts_budget_exceeded_runs(tmp_path):
    run_dir = tmp_path / "budget_exceeded"
    write_json(run_dir / "run_manifest.json", {
        "evidence_type": "measured", "method": "ZeroGS", "scene": "lego",
        "hard_vram_limit_mb": 4096, "seed": 0,
    })
    write_json(run_dir / "run_status.json", {
        "status": "budget_exceeded", "oom": False, "budget_exceeded": True,
    })
    summary = aggregate(collect_runs(tmp_path))
    assert summary[0]["budget_exceeded_runs"] == 1
