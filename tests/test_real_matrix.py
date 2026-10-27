from pathlib import Path

from experiments.run_real_matrix import build_run_commands, parse_scene


def test_parse_scene_and_build_commands(tmp_path):
    scene_path = tmp_path / "bonsai"
    scene_path.mkdir()
    name, parsed_path = parse_scene(f"bonsai={scene_path}")
    assert name == "bonsai"
    assert parsed_path == scene_path.resolve()

    repo = Path(__file__).resolve().parents[1]
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    commands = build_run_commands(
        repo, "bonsai", parsed_path, tmp_path / "run",
        4096.0, 2, 30_000, calibration,
    )
    assert len(commands) == 3
    assert "--cost_calibration" in commands[0]
    assert commands[0][commands[0].index("-i") + 1] == "images_4"
    assert "--eval" in commands[0] and "--eval" in commands[1]
    assert "--skip_train" in commands[1]
    assert commands[2][-2] == "-m"
