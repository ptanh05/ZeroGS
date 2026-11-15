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
            self.assertEqual(data["termination_signal"], "SIGTERM")
            self.assertTrue(data["graceful_termination"])
            print("[PASS] Test 2: Violation detected at iteration 600 and terminated via SIGTERM.")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
