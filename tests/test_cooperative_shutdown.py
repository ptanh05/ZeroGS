#!/usr/bin/env python3
"""
Unit Tests for ZeroGS Cooperative Shutdown Safety Hardening (Phase 5.5).

Covers all 6 Phase 5.5 acceptance criteria on CPU:
1. Stale stop.flag (prior PID or old mtime from previous session) does not abort new sessions.
2. Startup stop request targeting current process PID is preserved and honored.
3. Emergency checkpoint is written atomically (.tmp -> os.replace) and verifies via torch.load.
4. Periodic milestone checkpoints are never overwritten by emergency checkpoints.
5. Orphaned .tmp files are not recognized as valid checkpoints and are cleaned at startup.
6. Watchdog uncooperative timeout escalates to TerminateProcess and never claims graceful shutdown.
7. Full end-to-end integration: Watchdog detects breach -> cooperative shutdown -> verified checkpoint.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import unittest
import subprocess
import zipfile
from pathlib import Path

import torch


PYTHON_EXE = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_SCRIPT = str(PROJECT_ROOT / "train.py")
WATCHDOG_SCRIPT = str(PROJECT_ROOT / "scripts" / "vram_watchdog.py")


class TestCooperativeShutdownSafetyHardening(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="zerogs_phase55_test_")
        self.model_dir = Path(self.test_dir) / "model_output"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.stop_flag = self.model_dir / "stop.flag"
        self.telemetry_path = self.model_dir / "telemetry.jsonl"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_stale_flag_does_not_abort_new_session(self):
        """Acceptance 1: Stale stop.flag with prior PID is removed; new training session runs unaffected."""
        print("\n--- Running Test 1: Stale Stop Flag Safe Initialization ---")
        stale_payload = {
            "reason": "VRAM_LIMIT_EXCEEDED",
            "target_pid": 9999999,  # Non-existent / previous process PID
            "timestamp_utc": "2026-01-01T00:00:00Z"
        }
        with open(self.stop_flag, "w", encoding="utf-8") as f:
            json.dump(stale_payload, f)

        res = subprocess.run([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "5",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)

        self.assertEqual(res.returncode, 0, f"train.py should exit with 0. Stderr: {res.stderr}")
        self.assertIn("Removed stale stop flag", res.stdout)
        self.assertIn("Training completed successfully", res.stdout)
        self.assertNotIn("Cooperative shutdown requested", res.stdout)

        # Confirm no emergency stop artifacts created
        manifest_path = self.model_dir / "cooperative_shutdown.json"
        self.assertFalse(manifest_path.exists(), "Cooperative shutdown manifest should NOT exist for completed run")
        stop_chkpnts = list(self.model_dir.glob("chkpnt_stop_*.pth"))
        self.assertEqual(len(stop_chkpnts), 0, "No stop checkpoints should exist")
        print("[PASS] Acceptance 1: Stale flag safely purged; new training session completed uninterrupted.")

    def test_02_startup_stop_request_preserved_and_handled(self):
        """Acceptance 2: Stop request targeting current PID during startup is preserved and cleanly handled."""
        print("\n--- Running Test 2: Startup Race & Targeted Stop Request ---")
        proc = subprocess.Popen([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "20",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            # Immediately write targeted stop.flag matching proc.pid
            targeted_payload = {
                "reason": "EARLY_STARTUP_INTERCEPT",
                "target_pid": proc.pid,
                "timestamp_utc": "2026-11-16T08:00:00Z"
            }
            with open(self.stop_flag, "w", encoding="utf-8") as f:
                json.dump(targeted_payload, f)

            stdout, stderr = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0, f"Process should exit 0 on clean stop. Stderr: {stderr}")
            self.assertIn("Cooperative shutdown requested at iteration 1", stdout)
            self.assertIn("Reason: EARLY_STARTUP_INTERCEPT", stdout)

            # Verify manifest
            manifest_path = self.model_dir / "cooperative_shutdown.json"
            self.assertTrue(manifest_path.exists())
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            self.assertEqual(manifest["status"], "COOPERATIVE_SHUTDOWN_COMPLETED")
            self.assertEqual(manifest["stop_reason"], "EARLY_STARTUP_INTERCEPT")
            self.assertEqual(manifest["final_iteration"], 1)
            print("[PASS] Acceptance 2: Targeted startup stop request preserved without race loss.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_03_atomic_emergency_checkpoint_and_loadability(self):
        """Acceptance 3: Emergency checkpoint written atomically via .tmp -> os.replace and loadable via torch.load."""
        print("\n--- Running Test 3: Atomic Emergency Checkpoint & Loadability ---")
        proc = subprocess.Popen([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "10",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            targeted_payload = {
                "reason": "VRAM_LIMIT_EXCEEDED",
                "target_pid": proc.pid
            }
            with open(self.stop_flag, "w", encoding="utf-8") as f:
                json.dump(targeted_payload, f)

            stdout, stderr = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0, f"train.py should exit 0. Stderr: {stderr}")
            self.assertIn("Saved cooperative shutdown checkpoint atomically:", stdout)
            self.assertIn("Verified emergency checkpoint integrity and loadability", stdout)

            # Verify finalized checkpoint exists
            stop_chkpnt = self.model_dir / "chkpnt_stop_1.pth"
            self.assertTrue(stop_chkpnt.exists())
            self.assertGreater(stop_chkpnt.stat().st_size, 1000)

            # Verify no .tmp files remain
            tmp_files = list(self.model_dir.glob("chkpnt*.tmp*"))
            self.assertEqual(len(tmp_files), 0, f"No temporary checkpoint files should remain: {tmp_files}")

            # Independent torch.load validation
            loaded_data, iter_num = torch.load(stop_chkpnt, weights_only=False)
            self.assertEqual(iter_num, 1)
            self.assertIn("xyz", loaded_data)
            self.assertIn("scaling", loaded_data)
            self.assertIn("rotation", loaded_data)

            # Check manifest recorded verification
            manifest_path = self.model_dir / "cooperative_shutdown.json"
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            self.assertTrue(manifest["checkpoint_verified"])
            print("[PASS] Acceptance 3: Emergency checkpoint atomically persisted and loaded 100% intact.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_04_periodic_checkpoint_not_overwritten(self):
        """Acceptance 4: Periodic milestone checkpoint (chkpnt1.pth) is never overwritten by emergency checkpoint."""
        print("\n--- Running Test 4: Periodic Checkpoint Collision Protection ---")
        regular_chkpnt = self.model_dir / "chkpnt1.pth"
        sentinel_payload = {"mock_regular": True, "magic_number": 123456789}
        with open(regular_chkpnt, "wb") as f:
            torch.save((sentinel_payload, 1), f)
        original_size = regular_chkpnt.stat().st_size

        proc = subprocess.Popen([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "5",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            targeted_payload = {
                "reason": "TEST_COLLISION_AVOIDANCE",
                "target_pid": proc.pid
            }
            with open(self.stop_flag, "w", encoding="utf-8") as f:
                json.dump(targeted_payload, f)

            stdout, stderr = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0)

            # Verify regular checkpoint was not touched
            self.assertTrue(regular_chkpnt.exists())
            self.assertEqual(regular_chkpnt.stat().st_size, original_size)
            loaded_regular, _ = torch.load(regular_chkpnt, weights_only=False)
            self.assertEqual(loaded_regular.get("magic_number"), 123456789)

            # Verify emergency checkpoint has dedicated distinct name
            stop_chkpnt = self.model_dir / "chkpnt_stop_1.pth"
            self.assertTrue(stop_chkpnt.exists())
            self.assertNotEqual(stop_chkpnt.name, regular_chkpnt.name)
            print("[PASS] Acceptance 4: Regular milestone checkpoint completely preserved without collision.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_05_leftover_tmp_files_isolated_and_cleaned(self):
        """Acceptance 5: Leftover .tmp files are not treated as valid checkpoints and are cleaned at startup."""
        print("\n--- Running Test 5: Temporary File Isolation & Startup Cleanup ---")
        from scripts.vram_watchdog import verify_checkpoints_integrity

        # 1. Create a valid checkpoint
        valid_chkpnt = self.model_dir / "chkpnt1000.pth"
        with zipfile.ZipFile(str(valid_chkpnt), "w") as zf:
            zf.writestr("archive/data.pkl", b"valid_checkpoint_data_bytes_payload")

        # 2. Create orphaned .tmp files (simulating interrupted write)
        orphan_tmp1 = self.model_dir / "chkpnt_stop_500.pth.tmp_deadbeef"
        orphan_tmp2 = self.model_dir / "chkpnt_stop_600.pth.tmp_00112233"
        with open(orphan_tmp1, "wb") as f:
            f.write(b"truncated_garbage_bytes")
        with open(orphan_tmp2, "wb") as f:
            pass  # 0-byte orphan

        # 3. Verify watchdog integrity scanner ignores .tmp and counts only .pth
        report = verify_checkpoints_integrity(self.model_dir)
        self.assertEqual(report["count"], 1, "Only chkpnt1000.pth should be counted as a checkpoint")
        self.assertEqual(report["orphaned_tmp_count"], 2, "Orphaned .tmp files must be identified separately")
        self.assertTrue(report["all_valid"], "Valid checkpoints collection should remain all_valid=True")
        self.assertIn("chkpnt_stop_500.pth.tmp_deadbeef", report["orphaned_tmp_files"])

        # 4. Run train.py and verify startup cleanup removes orphan .tmp files
        res = subprocess.run([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "2",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)

        self.assertEqual(res.returncode, 0)
        self.assertIn("Cleaned up orphaned temporary checkpoint: chkpnt_stop_500.pth.tmp_deadbeef", res.stdout)
        self.assertIn("Cleaned up orphaned temporary checkpoint: chkpnt_stop_600.pth.tmp_00112233", res.stdout)

        # Confirm orphans are gone from disk
        self.assertFalse(orphan_tmp1.exists())
        self.assertFalse(orphan_tmp2.exists())
        self.assertTrue(valid_chkpnt.exists(), "Valid checkpoint must remain untouched")
        print("[PASS] Acceptance 5: Temporary files safely isolated from valid checkpoints and purged on startup.")

    def test_06_watchdog_uncooperative_timeout_never_claims_graceful(self):
        """Acceptance 6: When process ignores stop.flag, watchdog escalates to TerminateProcess and marks graceful=False."""
        print("\n--- Running Test 6: Uncooperative Timeout & Truthful Reporting ---")
        intercept_log = self.model_dir / "watchdog_intercept.json"

        # Spawn an uncooperative dummy process that ignores stop.flag
        dummy_code = f"""
import time, json
with open(r"{self.telemetry_path}", "w") as f:
    f.write(json.dumps({{"event": "start"}}) + "\\n")
    f.flush()
    # Write breach record
    f.write(json.dumps({{"event": "mutation", "iteration": 50, "memory_after": {{"allocated_mb": 5000.0, "reserved_mb": 5000.0, "conservative_footprint_mb": 5000.0}}}}) + "\\n")
    f.flush()
# Ignore stop.flag and sleep
time.sleep(20)
"""
        uncoop_proc = subprocess.Popen([PYTHON_EXE, "-c", dummy_code])
        try:
            # Wait for telemetry file to appear
            for _ in range(50):
                if self.telemetry_path.exists():
                    break
                time.sleep(0.1)

            # Run watchdog with short 1.5s cooperative timeout
            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(uncoop_proc.pid),
                "--telemetry_path", str(self.telemetry_path),
                "--limit_mb", "4096.0",
                "--allowance_mb", "256.0",
                "--poll_interval", "0.1",
                "--cooperative_timeout", "1.5",
                "--sigterm_timeout", "1.5",
                "--output_log", str(intercept_log)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)

            uncoop_proc.wait(timeout=5)
            self.assertEqual(wd_res.returncode, 0)
            self.assertTrue(intercept_log.exists())

            with open(intercept_log, "r", encoding="utf-8") as f:
                idata = json.load(f)

            self.assertEqual(idata["status"], "BUDGET_BREACH_INTERCEPTED")
            if sys.platform == "win32":
                self.assertEqual(idata["termination_signal"], "TerminateProcess")
                self.assertFalse(idata["graceful_termination"], "Watchdog MUST NOT claim graceful termination after TerminateProcess!")
                self.assertIn("abruptly", idata["windows_safety_limitation"])
            print("[PASS] Acceptance 6: Watchdog forced timeout correctly reports graceful_termination=False.")
        finally:
            if uncoop_proc.poll() is None:
                uncoop_proc.kill()

    def test_07_watchdog_cooperative_integration_e2e(self):
        """End-to-End: Watchdog detects breach -> cooperative shutdown -> verified checkpoint & graceful=True."""
        print("\n--- Running Test 7: Watchdog & Training Cooperative Integration E2E ---")
        intercept_log = self.model_dir / "watchdog_intercept.json"

        train_proc = subprocess.Popen([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "25",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ])

        try:
            for _ in range(50):
                if self.telemetry_path.exists():
                    break
                time.sleep(0.1)

            time.sleep(0.2)
            # Inject budget breach record
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                violating_rec = {
                    "event": "mutation_cycle",
                    "iteration": 2,
                    "memory_after": {
                        "allocated_mb": 200.0,
                        "reserved_mb": 350.0,
                        "conservative_footprint_mb": 350.0
                    }
                }
                f.write(json.dumps(violating_rec) + "\n")
                f.flush()

            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(train_proc.pid),
                "--telemetry_path", str(self.telemetry_path),
                "--limit_mb", "300.0",
                "--allowance_mb", "50.0",
                "--poll_interval", "0.1",
                "--cooperative_timeout", "5.0",
                "--output_log", str(intercept_log)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)

            train_proc.wait(timeout=5)
            self.assertEqual(train_proc.returncode, 0, "train.py must exit 0 on clean cooperative shutdown")
            self.assertEqual(wd_res.returncode, 0)

            self.assertTrue(intercept_log.exists())
            with open(intercept_log, "r", encoding="utf-8") as f:
                idata = json.load(f)

            self.assertEqual(idata["status"], "BUDGET_BREACH_INTERCEPTED")
            self.assertEqual(idata["termination_signal"], "COOPERATIVE_STOP")
            self.assertTrue(idata["graceful_termination"])
            self.assertTrue(idata["telemetry_flushed"])
            self.assertIsNotNone(idata["cooperative_shutdown_manifest"])
            self.assertTrue(idata["cooperative_shutdown_manifest"]["checkpoint_verified"])
            print("[PASS] Test 7: Full E2E cooperative interception with verified checkpoint & graceful=True.")
        finally:
            if train_proc.poll() is None:
                train_proc.kill()

    def test_08_checkpoint_verification_both_schemas_dict_and_tuple(self):
        """Test 8 (Regression MAJOR-1/2): verify_checkpoint_payload supports both Mock dict and GaussianModel tuple."""
        print("\n--- Running Test 8: Checkpoint Verification for Both Dict and Tuple Schemas ---")
        from train import verify_checkpoint_payload
        from scene.gaussian_model import GaussianModel
        from arguments import OptimizationParams
        from argparse import ArgumentParser

        # 1. Simulation dict schema
        self.assertTrue(verify_checkpoint_payload({"xyz": torch.zeros(4, 3)}, 10, 10))
        self.assertFalse(verify_checkpoint_payload({"xyz": None}, 10, 10))
        self.assertFalse(verify_checkpoint_payload({"xyz": torch.zeros(4, 3)}, 9, 10))

        # 2. Real GaussianModel.capture() tuple schema on CPU
        parser = ArgumentParser()
        op = OptimizationParams(parser)
        opt_args = op.extract(parser.parse_args([]))

        gm = GaussianModel(sh_degree=3)
        gm._xyz = torch.nn.Parameter(torch.randn(8, 3))
        gm._features_dc = torch.nn.Parameter(torch.randn(8, 1, 3))
        gm._features_rest = torch.nn.Parameter(torch.randn(8, 15, 3))
        gm._opacity = torch.nn.Parameter(torch.randn(8, 1))
        gm._scaling = torch.nn.Parameter(torch.randn(8, 3))
        gm._rotation = torch.nn.Parameter(torch.randn(8, 4))
        gm.max_radii2D = torch.zeros(8)
        gm.spatial_lr_scale = 1.0
        gm.birth_iteration = torch.ones(8, dtype=torch.int32)
        gm.training_setup(opt_args)

        (gm._xyz ** 2).sum().backward()
        gm.optimizer.step()

        chkpnt_path = self.model_dir / "chkpnt_stop_7.pth"
        torch.save((gm.capture(), 7), chkpnt_path)
        loaded_data, loaded_iter = torch.load(chkpnt_path, map_location="cpu", weights_only=False)

        self.assertIsInstance(loaded_data, tuple)
        self.assertGreaterEqual(len(loaded_data), 12)
        self.assertTrue(verify_checkpoint_payload(loaded_data, loaded_iter, 7))
        # Short tuple or missing _xyz at index 1 must fail
        self.assertFalse(verify_checkpoint_payload(loaded_data[:11], 7, 7))
        bad_tuple = list(loaded_data)
        bad_tuple[1] = None
        self.assertFalse(verify_checkpoint_payload(tuple(bad_tuple), 7, 7))
        print("[PASS] Test 8: Both simulation dict and real GaussianModel tuple schemas verified.")

    def test_09_run_status_distinction_and_stale_manifest_archived(self):
        """Test 9 (Regression MINOR-4): run_status distinguishes cooperative_shutdown vs completed; stale manifest archived."""
        print("\n--- Running Test 9: run_status Distinction & Stale Manifest Archival ---")
        manifest_path = self.model_dir / "cooperative_shutdown.json"
        run_status_path = self.model_dir / "run_status.json"

        # Pre-seed stale cooperative_shutdown.json from prior session
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"status": "COOPERATIVE_SHUTDOWN_COMPLETED", "final_iteration": 999}, f)

        # 1. Normal completion run
        res = subprocess.run([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "3",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Archived stale cooperative shutdown manifest", res.stdout)
        self.assertFalse(manifest_path.exists(), "Stale cooperative_shutdown.json must be archived away")
        self.assertGreaterEqual(len(list(self.model_dir.glob("cooperative_shutdown.stale_*.json"))), 1)

        with open(run_status_path, "r", encoding="utf-8") as f:
            status_completed = json.load(f)
        self.assertEqual(status_completed["status"], "completed")

        # 2. Cooperative shutdown run
        proc = subprocess.Popen([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "10",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            with open(self.stop_flag, "w", encoding="utf-8") as f:
                json.dump({"reason": "STATUS_DISTINCTION_TEST", "target_pid": proc.pid}, f)
            stdout, stderr = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0, f"Stderr: {stderr}")
            with open(run_status_path, "r", encoding="utf-8") as f:
                status_coop = json.load(f)
            self.assertEqual(status_coop["status"], "cooperative_shutdown")
            self.assertEqual(status_coop["final_iteration"], 1)
            print("[PASS] Test 9: run_status accurately distinguishes cooperative_shutdown vs completed.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_10_pidless_stop_flag_timing_startup_vs_stale(self):
        """Test 10 (Regression MINOR-5): PID-less stop flag after Popen (during import) is preserved; old mtime is stale."""
        print("\n--- Running Test 10: PID-less Stop Flag Startup Timing vs Stale ---")
        # 1. Old PID-less stop.flag (mtime 15s in the past) must be purged as stale
        with open(self.stop_flag, "w", encoding="utf-8") as f:
            f.write("MANUAL_OLD_STOP")
        old_ts = time.time() - 15.0
        os.utime(self.stop_flag, (old_ts, old_ts))

        res = subprocess.run([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "3",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Removed stale stop flag", res.stdout)
        self.assertIn("Training completed successfully", res.stdout)

        # 2. PID-less stop.flag created right after process spawn (before torch import finishes) must NOT be deleted
        proc = subprocess.Popen([
            PYTHON_EXE, TRAIN_SCRIPT,
            "--simulation_mode",
            "-m", str(self.model_dir),
            "--iterations", "10",
            "--data_device", "cpu",
            "--telemetry_jsonl", str(self.telemetry_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            with open(self.stop_flag, "w", encoding="utf-8") as f:
                f.write("IMMEDIATE_PIDLESS_STOP")
            stdout, stderr = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0, f"Stderr: {stderr}")
            self.assertIn("Cooperative shutdown requested at iteration 1", stdout)
            self.assertIn("Reason: IMMEDIATE_PIDLESS_STOP", stdout)
            print("[PASS] Test 10: _PROCESS_START_TIME preserves immediate PID-less flags and rejects old ones.")
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main(verbosity=2)
