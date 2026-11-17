#!/usr/bin/env python3
"""
Automated CPU-Only Test Suite for ZeroGS VRAM Watchdog (Phase 5.3).

Executes 6 deterministic test scenarios on CPU without initializing CUDA:
1. No violation: Watchdog does not kill process; both exit cleanly.
2. Telemetry exceeds threshold: Watchdog detects breach and sends SIGTERM.
3. Process does not respond to SIGTERM: Watchdog escalates to SIGKILL.
4. JSONL has malformed/partial lines: Watchdog parses robustly without crashing.
5. Training terminates before any violation: Watchdog records summary and exits cleanly.
6. Invalid PID: Watchdog detects non-existent PID and reports error immediately.
"""

import os
import sys
import time
import json
import signal
import shutil
import tempfile
import unittest
import subprocess
from pathlib import Path


PYTHON_EXE = sys.executable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
WATCHDOG_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "vram_watchdog.py")


class TestVramWatchdogCpuOnly(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="zerogs_watchdog_test_")
        self.telemetry_file = Path(self.test_dir) / "telemetry.jsonl"
        self.intercept_file = Path(self.test_dir) / "watchdog_intercept.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_no_violation_process_completes_cleanly(self):
        """Test 1: Normal execution without budget breach - watchdog does not kill process."""
        print("\n--- Running Test 1: No Violation ---")
        dummy_code = f"""
import time, json
with open(r"{self.telemetry_file}", "w") as f:
    f.write(json.dumps({{"event": "training_start", "initial_gaussians": 1000}}) + "\\n")
    f.flush()
    time.sleep(0.2)
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 500, "memory_after": {{"allocated_mb": 1000.0, "reserved_mb": 2000.0, "conservative_footprint_mb": 2000.0}}}}) + "\\n")
    f.flush()
    time.sleep(0.2)
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 600, "memory_after": {{"allocated_mb": 1100.0, "reserved_mb": 2200.0, "conservative_footprint_mb": 2200.0}}}}) + "\\n")
    f.flush()
    time.sleep(0.2)
"""
        proc = subprocess.Popen([PYTHON_EXE, "-c", dummy_code])
        try:
            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(proc.pid),
                "--telemetry_path", str(self.telemetry_file),
                "--limit_mb", "4096.0",
                "--allowance_mb", "256.0",
                "--poll_interval", "0.1",
                "--output_log", str(self.intercept_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)

            proc.wait(timeout=5)
            self.assertEqual(proc.returncode, 0, "Dummy process should exit with returncode 0")
            self.assertEqual(wd_res.returncode, 0, f"Watchdog should exit with 0. Stderr: {wd_res.stderr}")
            self.assertFalse(self.intercept_file.exists(), "Intercept log should NOT be created for clean run")
            self.assertIn("Clean execution within budget", wd_res.stdout)
            print("[PASS] Test 1: Completed cleanly without premature termination.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_02_telemetry_exceeds_threshold_sigterm_sent(self):
        """Test 2: Telemetry exceeds threshold - watchdog detects and sends SIGTERM."""
        print("\n--- Running Test 2: Threshold Violation & SIGTERM ---")
        dummy_code = f"""
import time, json
with open(r"{self.telemetry_file}", "w") as f:
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 500, "memory_after": {{"allocated_mb": 1000.0, "reserved_mb": 2000.0, "conservative_footprint_mb": 2000.0}}}}) + "\\n")
    f.flush()
    time.sleep(0.2)
    # Violating record: compound footprint = 3900 + 256 = 4156 MB > 4096 MB
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 600, "memory_after": {{"allocated_mb": 3500.0, "reserved_mb": 3900.0, "conservative_footprint_mb": 3900.0}}}}) + "\\n")
    f.flush()

# Stay alive until terminated by watchdog
time.sleep(15)
"""
        proc = subprocess.Popen([PYTHON_EXE, "-c", dummy_code])
        try:
            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(proc.pid),
                "--telemetry_path", str(self.telemetry_file),
                "--limit_mb", "4096.0",
                "--allowance_mb", "256.0",
                "--poll_interval", "0.1",
                "--sigterm_timeout", "3.0",
                "--output_log", str(self.intercept_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)

            proc.wait(timeout=5)
            self.assertNotEqual(proc.poll(), None, "Dummy process should be terminated")
            self.assertEqual(wd_res.returncode, 0, f"Watchdog should exit with 0 after intercept. Stderr: {wd_res.stderr}")
            self.assertTrue(self.intercept_file.exists(), "watchdog_intercept.json must exist")

            with open(self.intercept_file, "r") as f:
                data = json.load(f)

            self.assertEqual(data["status"], "BUDGET_BREACH_INTERCEPTED")
            self.assertEqual(data["first_observed_iteration"], 600)
            self.assertGreater(data["observed_compound_footprint_mb"], 4096.0)
            if sys.platform == "win32":
                self.assertEqual(data["termination_signal"], "TerminateProcess")
                self.assertFalse(data["graceful_termination"], "On Windows, TerminateProcess is non-graceful")
                self.assertIn("abrupt", data.get("windows_safety_limitation", "").lower())
            else:
                self.assertEqual(data["termination_signal"], "SIGTERM")
                self.assertTrue(data["graceful_termination"])
            self.assertTrue(data.get("telemetry_flushed", False), "Telemetry should be flushed")
            print(f"[PASS] Test 2: Violation detected at iteration 600 and terminated via {data['termination_signal']}.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_03_process_unresponsive_escalates_to_sigkill(self):
        """Test 3: Process ignores SIGTERM - watchdog escalates to SIGKILL."""
        print("\n--- Running Test 3: Unresponsive Process & SIGKILL Escalation ---")
        if sys.platform == "win32":
            print("[SKIP] Test 3: Native SIGKILL escalation test requires POSIX environment (run inside WSL2).")
            return

        dummy_code = f"""
import time, json, signal
# Explicitly ignore SIGTERM
signal.signal(signal.SIGTERM, signal.SIG_IGN)

with open(r"{self.telemetry_file}", "w") as f:
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 700, "memory_after": {{"allocated_mb": 4100.0, "reserved_mb": 4200.0, "conservative_footprint_mb": 4200.0}}}}) + "\\n")
    f.flush()

# Keep running despite SIGTERM
start = time.time()
while time.time() - start < 30:
    time.sleep(0.1)
"""
        proc = subprocess.Popen([PYTHON_EXE, "-c", dummy_code])
        try:
            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(proc.pid),
                "--telemetry_path", str(self.telemetry_file),
                "--limit_mb", "4096.0",
                "--allowance_mb", "256.0",
                "--poll_interval", "0.1",
                "--sigterm_timeout", "1.0",  # Short timeout for quick escalation
                "--output_log", str(self.intercept_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)

            proc.wait(timeout=5)
            self.assertNotEqual(proc.poll(), None, "Dummy process should be killed")
            self.assertTrue(self.intercept_file.exists(), "watchdog_intercept.json must exist")

            with open(self.intercept_file, "r") as f:
                data = json.load(f)

            self.assertEqual(data["termination_signal"], "SIGKILL")
            print("[PASS] Test 3: Unresponsive process successfully escalated to SIGKILL.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_04_jsonl_malformed_lines_handled_without_crash(self):
        """Test 4: JSONL has malformed, corrupted, or incomplete lines - watchdog survives."""
        print("\n--- Running Test 4: Malformed/Partial JSON Resilience ---")
        dummy_code = f"""
import time, json
with open(r"{self.telemetry_file}", "w") as f:
    f.write('{{"event": "incomp')  # Incomplete line without newline
    f.flush()
    time.sleep(0.2)
    f.write('lete_write_without_newline')
    f.flush()
    time.sleep(0.2)
    f.write('\\n')  # Completed invalid line
    f.flush()
    time.sleep(0.2)
    f.write('{{corrupted json syntax!}}\\n')  # Corrupted JSON
    f.flush()
    time.sleep(0.2)
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 500, "memory_after": {{"allocated_mb": 800.0, "reserved_mb": 1500.0, "conservative_footprint_mb": 1500.0}}}}) + "\\n")
    f.flush()
    time.sleep(0.2)
"""
        proc = subprocess.Popen([PYTHON_EXE, "-c", dummy_code])
        try:
            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(proc.pid),
                "--telemetry_path", str(self.telemetry_file),
                "--limit_mb", "4096.0",
                "--poll_interval", "0.1",
                "--output_log", str(self.intercept_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)

            proc.wait(timeout=5)
            self.assertEqual(wd_res.returncode, 0, f"Watchdog should not crash on malformed lines. Stderr: {wd_res.stderr}")
            self.assertIn("Clean execution within budget", wd_res.stdout)
            print("[PASS] Test 4: Malformed and partial lines handled gracefully without crash.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_05_training_terminates_before_violation(self):
        """Test 5: Training finishes normally before any violation - watchdog exits with 0."""
        print("\n--- Running Test 5: Clean Training Termination ---")
        dummy_code = f"""
import time, json
with open(r"{self.telemetry_file}", "w") as f:
    f.write(json.dumps({{"event": "training_start", "initial_gaussians": 138000}}) + "\\n")
    f.flush()
    time.sleep(0.2)
    f.write(json.dumps({{"event": "mutation_cycle", "iteration": 1000, "memory_after": {{"allocated_mb": 1200.0, "reserved_mb": 2500.0, "conservative_footprint_mb": 2500.0}}}}) + "\\n")
    f.flush()
    time.sleep(0.2)
    f.write(json.dumps({{"event": "training_end", "final_gaussians": 190000, "memory": {{"allocated_mb": 1250.0, "reserved_mb": 2550.0, "conservative_footprint_mb": 2550.0}}}}) + "\\n")
    f.flush()
    time.sleep(0.2)
"""
        proc = subprocess.Popen([PYTHON_EXE, "-c", dummy_code])
        try:
            wd_res = subprocess.run([
                PYTHON_EXE, WATCHDOG_SCRIPT,
                "--pid", str(proc.pid),
                "--telemetry_path", str(self.telemetry_file),
                "--limit_mb", "4096.0",
                "--poll_interval", "0.1",
                "--output_log", str(self.intercept_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)

            proc.wait(timeout=5)
            self.assertEqual(wd_res.returncode, 0)
            self.assertIn("Surveillance Summary", wd_res.stdout)
            self.assertIn("Target process", wd_res.stdout)
            self.assertFalse(self.intercept_file.exists())
            print("[PASS] Test 5: Clean training termination logged and watchdog exited with 0.")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_06_invalid_or_nonexistent_pid(self):
        """Test 6: Invalid or non-existent PID - watchdog reports error and exits immediately."""
        print("\n--- Running Test 6: Invalid PID Handling ---")
        invalid_pid = 99999999
        wd_res = subprocess.run([
            PYTHON_EXE, WATCHDOG_SCRIPT,
            "--pid", str(invalid_pid),
            "--telemetry_path", str(self.telemetry_file),
            "--limit_mb", "4096.0",
            "--poll_interval", "0.1",
            "--output_log", str(self.intercept_file)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

        self.assertEqual(wd_res.returncode, 1, "Watchdog should exit with 1 on invalid PID")
        self.assertIn("[WATCHDOG ERROR] Target PID", wd_res.stdout)
        self.assertIn("is not running or invalid", wd_res.stdout)
        print("[PASS] Test 6: Invalid PID detected immediately with code 1.")

    def test_07_checkpoint_and_telemetry_integrity(self):
        """Test 7: Verify checkpoint archive validity and telemetry flush verification."""
        print("\n--- Running Test 7: Checkpoint and Telemetry Integrity Verification ---")
        import zipfile
        from scripts.vram_watchdog import verify_checkpoints_integrity, verify_telemetry_flushed

        test_dir = Path(self.test_dir)
        valid_cp = test_dir / "chkpnt1000.pth"
        corrupt_cp = test_dir / "chkpnt2000.pth"
        zero_byte_cp = test_dir / "chkpnt3000.pth"

        # 1. Create a valid mock PyTorch checkpoint (valid zip archive)
        with zipfile.ZipFile(str(valid_cp), "w") as zf:
            zf.writestr("archive/data.pkl", b"mock_pytorch_tensor_data_payload_bytes")

        # 2. Create a corrupted zip
        with open(corrupt_cp, "wb") as f:
            f.write(b"PK\x03\x04corrupted_header_data_not_a_valid_zip_archive")

        # 3. Create a 0-byte file (truncated write)
        with open(zero_byte_cp, "wb") as f:
            pass

        report = verify_checkpoints_integrity(test_dir)
        self.assertEqual(report["count"], 3)
        self.assertFalse(report["all_valid"])

        cp_by_name = {c["filename"]: c for c in report["checkpoints"]}
        self.assertTrue(cp_by_name["chkpnt1000.pth"]["valid"])
        self.assertEqual(cp_by_name["chkpnt1000.pth"]["status"], "OK")

        self.assertFalse(cp_by_name["chkpnt2000.pth"]["valid"])
        self.assertFalse(cp_by_name["chkpnt3000.pth"]["valid"])
        self.assertEqual(cp_by_name["chkpnt3000.pth"]["status"], "ZERO_BYTES")

        # Test telemetry flush verification
        with open(self.telemetry_file, "w", encoding="utf-8") as f:
            f.write('{"event": "mutation", "iter": 500}\n')
        flushed, msg = verify_telemetry_flushed(self.telemetry_file)
        self.assertTrue(flushed)
        self.assertEqual(msg, "TELEMETRY_FLUSHED_VALID")

        # Truncated telemetry line without newline
        with open(self.telemetry_file, "a", encoding="utf-8") as f:
            f.write('{"event": "unfin')
        flushed_corrupt, msg_corrupt = verify_telemetry_flushed(self.telemetry_file)
        self.assertFalse(flushed_corrupt)
        print("[PASS] Test 7: Checkpoint archive integrity and telemetry flush status verified.")

    def test_08_verify_telemetry_flushed_long_line_over_4096_bytes(self):
        """Test 8 (Regression MINOR-6): verify_telemetry_flushed handles JSONL lines > 4096 bytes."""
        print("\n--- Running Test 8: Long JSONL Line (>4096 bytes) Telemetry Verification ---")
        from scripts.vram_watchdog import verify_telemetry_flushed

        long_payload = "x" * 16000
        with open(self.telemetry_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "start", "iteration": 0}) + "\n")
            f.write(json.dumps({"event": "mutation_cycle", "iteration": 100, "blob": long_payload}) + "\n")

        flushed, msg = verify_telemetry_flushed(self.telemetry_file)
        self.assertTrue(flushed, f"Expected TELEMETRY_FLUSHED_VALID for >16KB last line, got: {msg}")
        self.assertEqual(msg, "TELEMETRY_FLUSHED_VALID")

        # Truncated >4KB line must still be rejected as corrupt
        with open(self.telemetry_file, "a", encoding="utf-8") as f:
            f.write('{"event": "partial", "blob": "' + ("y" * 6000))
        flushed_bad, msg_bad = verify_telemetry_flushed(self.telemetry_file)
        self.assertFalse(flushed_bad)
        self.assertIn("TELEMETRY_CORRUPT", msg_bad)
        print("[PASS] Test 8: Telemetry verification handles >16KB lines cleanly.")

    def test_09_atomic_stop_flag_and_physical_gpu_limit_cooperative_first(self):
        """Test 9 (Regression MINOR-3 & MINOR-7): Atomic stop.flag write + physical GPU breach tries cooperative stop first."""
        print("\n--- Running Test 9: Atomic stop.flag & Physical GPU Cooperative-First ---")
        from unittest.mock import patch
        from scripts.vram_watchdog import write_json_atomic, main as watchdog_main

        stop_flag = Path(self.test_dir) / "stop.flag"
        write_json_atomic(stop_flag, {"reason": "TEST_ATOMIC", "target_pid": 42})
        self.assertTrue(stop_flag.exists())
        self.assertEqual(list(Path(self.test_dir).glob("stop.flag.tmp_*")), [])
        with open(stop_flag, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["target_pid"], 42)
        stop_flag.unlink()

        # Subprocess watches stop.flag and exits cooperatively with exit code 0
        coop_code = f"""
import time, json
from pathlib import Path
with open(r"{self.telemetry_file}", "w") as f:
    f.write(json.dumps({{"event": "training_start"}}) + "\\n")
    f.flush()
stop_path = Path(r"{stop_flag}")
for _ in range(100):
    if stop_path.exists():
        with open(r"{self.telemetry_file}", "a") as f:
            f.write(json.dumps({{"event": "cooperative_shutdown"}}) + "\\n")
            f.flush()
        break
    time.sleep(0.05)
"""
        proc = subprocess.Popen([PYTHON_EXE, "-c", coop_code])
        try:
            for _ in range(50):
                if self.telemetry_file.exists():
                    break
                time.sleep(0.05)

            argv_sim = [
                "vram_watchdog.py",
                "--pid", str(proc.pid),
                "--telemetry_path", str(self.telemetry_file),
                "--stop_file", str(stop_flag),
                "--limit_mb", "8192.0",
                "--physical_gpu_limit_mb", "4096.0",
                "--poll_interval", "0.1",
                "--cooperative_timeout", "5.0",
                "--output_log", str(self.intercept_file),
            ]
            with patch("sys.argv", argv_sim), \
                 patch("scripts.vram_watchdog.find_nvidia_smi", return_value="mock-smi"), \
                 patch("scripts.vram_watchdog.query_physical_gpu_memory", return_value=(6000.0, 8151.0)):
                with self.assertRaises(SystemExit) as cm:
                    watchdog_main()
                self.assertEqual(cm.exception.code, 0)

            proc.wait(timeout=5)
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(self.intercept_file.exists())
            self.assertEqual(list(Path(self.test_dir).glob("stop.flag.tmp_*")), [])
            with open(self.intercept_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "PHYSICAL_GPU_LIMIT_BREACHED")
            self.assertEqual(data["termination_signal"], "COOPERATIVE_STOP")
            self.assertTrue(data["graceful"])
            self.assertTrue(data["graceful_termination"])
            self.assertTrue(data["telemetry_flushed"])
            print("[PASS] Test 9: Atomic stop.flag & physical GPU limit cooperative shutdown verified.")
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main(verbosity=2)
