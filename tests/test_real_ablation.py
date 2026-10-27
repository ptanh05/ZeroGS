import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_real_ablation import ABLATIONS, build_ablation_commands


def test_ablation_commands_encode_real_variants(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    scene = tmp_path / "scene"
    scene.mkdir()
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")

    commands = build_ablation_commands(
        repo, "bonsai", scene, tmp_path / "run", 6144.0, 1, 30_000,
        calibration, "s1_full_no_protection",
    )
    assert "--disable_structure_protection" in commands[0]
    assert commands[0][commands[0].index("--occlusion_mode") + 1] == "full"
    assert commands[0][commands[0].index("-i") + 1] == "images_4"
    assert "--eval" in commands[0] and "--eval" in commands[1]
    assert set(ABLATIONS) >= {"p3_gain_per_byte", "p3_demand_only", "p3_random", "s1_full"}
