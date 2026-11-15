#!/usr/bin/env python3
"""
ZeroGS CPU-Side VRAM Watchdog Daemon (Phase 5.3).

Monitors training execution under a strict 4096 MiB budget by observing:
1. PyTorch allocator counters via append-only `telemetry.jsonl` (conservative footprint, reserved memory, compound footprint).
2. Auxiliary physical GPU memory via `nvidia-smi` (distinguishing OS/desktop usage from PyTorch allocations).

Enforces safe termination via graceful SIGTERM (with 5.0s timeout before SIGKILL escalation).

SCIENTIFIC DISCLAIMER:
This watchdog operates as an external, best-effort asynchronous monitor. Because 3DGS
densification writes telemetry at 100-iteration cycles, this watchdog cannot guarantee
sub-millisecond or hardware-enforced hard memory clamping. True first violation occurred
in iteration window [T_obs - 99, T_obs].
"""

import os
import sys
import time
import json
import signal
import shutil
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def find_nvidia_smi() -> Optional[str]:
    """Locate nvidia-smi across Linux, WSL2, and Windows paths."""
    candidates = [
        shutil.which("nvidia-smi"),
        "/usr/lib/wsl/lib/nvidia-smi",
        "/usr/bin/nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def query_physical_gpu_memory(smi_path: Optional[str]) -> Optional[Tuple[float, float]]:
    """Query total system-wide GPU used and total memory in MiB."""
    if not smi_path:
        return None
    try:
        res = subprocess.run(
            [smi_path, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            check=True
        )
        parts = res.stdout.strip().split(",")
        if len(parts) >= 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    return None


def is_process_alive(pid: int) -> bool:
    """
    Check if process with given PID exists, is accessible, and is actively running.
    Handles Unix zombie/defunct state where process has terminated but not yet reaped.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as err:
        import errno
        # ESRCH: no such process; EPERM: process exists but another user owns it (alive)
        return err.errno == errno.EPERM

    # On Linux/WSL2, inspect /proc/<pid>/status to verify process is not a defunct zombie
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.exists():
        try:
            with open(proc_status, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        state = line.split()[1]
                        # Z: zombie, X: dead
                        if state in ("Z", "X"):
                            return False
                        break
        except (IOError, OSError):
            return False

    return True


def terminate_process(pid: int, timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Send SIGTERM to target PID, wait up to timeout seconds, then escalate to SIGKILL.
    Returns (success, signal_used).
    """
    if not is_process_alive(pid):
        return True, "ALREADY_TERMINATED"

    print(f"[WATCHDOG] Sending SIGTERM to PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"[WATCHDOG] Failed to send SIGTERM: {e}")
        return not is_process_alive(pid), "SIGTERM_FAILED"

    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_process_alive(pid):
            print(f"[WATCHDOG] PID {pid} terminated gracefully via SIGTERM within {time.time() - start_time:.2f}s.")
            return True, "SIGTERM"
        time.sleep(0.1)

    print(f"[WATCHDOG] PID {pid} did not exit within {timeout}s grace period. Escalating to SIGKILL...")
    try:
        # SIGKILL is available on POSIX/Linux; on Windows SIGTERM is already TerminateProcess
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, sigkill)
    except OSError as e:
        print(f"[WATCHDOG] Error sending SIGKILL: {e}")

    time.sleep(0.5)
    alive = is_process_alive(pid)
    if not alive:
        print(f"[WATCHDOG] PID {pid} successfully terminated via SIGKILL.")
        return True, "SIGKILL"
    else:
        print(f"[WATCHDOG ERROR] PID {pid} could not be terminated!")
        return False, "SIGKILL_FAILED"


def extract_memory_from_record(rec: Dict[str, Any], allowance_mb: float) -> Optional[Dict[str, Any]]:
    """
    Extract standardized memory metrics from a telemetry JSON record.
    Distinguishes:
    - allocated_mb: PyTorch tensors
    - reserved_mb: PyTorch caching allocator pool
    - conservative_footprint_mb: max(allocated, reserved)
    - compound_footprint_mb: conservative_footprint_mb + allowance_mb
    """
    mem_dict = rec.get("memory_after") or rec.get("memory") or {}
    if not mem_dict and "current_allocated_mb" in rec:
        mem_dict = {
            "allocated_mb": rec.get("current_allocated_mb", 0.0),
            "reserved_mb": rec.get("current_reserved_mb", 0.0),
            "conservative_footprint_mb": rec.get("current_footprint_mb", 0.0),
        }

    if not mem_dict:
        return None

    allocated_mb = float(mem_dict.get("allocated_mb", 0.0))
    reserved_mb = float(mem_dict.get("reserved_mb", 0.0))
    conservative_mb = float(mem_dict.get("conservative_footprint_mb", max(allocated_mb, reserved_mb)))

    # Check if decision.stats has pre-computed compound footprint (ZeroGS runs)
    decision_stats = (rec.get("decision") or {}).get("stats", {})
    if "current_footprint_mb" in decision_stats:
        compound_footprint_mb = float(decision_stats["current_footprint_mb"])
    else:
        compound_footprint_mb = conservative_mb + allowance_mb

    iteration = rec.get("iteration", -1)

    return {
        "iteration": iteration,
        "event": rec.get("event", "unknown"),
        "allocated_mb": allocated_mb,
        "reserved_mb": reserved_mb,
        "conservative_footprint_mb": conservative_mb,
        "compound_footprint_mb": compound_footprint_mb,
        "raw_record": rec
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZeroGS Out-of-Process VRAM Watchdog Daemon (Phase 5.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pid", type=int, required=True, help="PID of the target training process")
    parser.add_argument("--telemetry_path", type=str, required=True, help="Path to telemetry.jsonl")
    parser.add_argument("--limit_mb", type=float, default=4096.0, help="VRAM budget ceiling in MiB")
    parser.add_argument("--allowance_mb", type=float, default=256.0,
                        help="Non-PyTorch allowance to add to PyTorch conservative footprint (matches ZeroGS admission formula)")
    parser.add_argument("--poll_interval", type=float, default=0.25, help="Polling interval in seconds")
    parser.add_argument("--sigterm_timeout", type=float, default=5.0, help="Seconds to wait after SIGTERM before SIGKILL")
    parser.add_argument("--output_log", type=str, default=None,
                        help="Destination for watchdog_intercept.json (defaults to <telemetry_dir>/watchdog_intercept.json)")
    parser.add_argument("--file_wait_timeout", type=float, default=60.0,
                        help="Max seconds to wait for telemetry file creation upon process start")
    parser.add_argument("--physical_gpu_limit_mb", type=float, default=0.0,
                        help="Optional emergency limit on system-wide physical GPU VRAM (0.0 to disable)")
    return parser.parse_args()


def main():
    args = parse_args()
    telem_path = Path(args.telemetry_path)
    output_log = Path(args.output_log) if args.output_log else telem_path.parent / "watchdog_intercept.json"
    smi_path = find_nvidia_smi()

    print("=" * 70)
    print("ZeroGS VRAM Watchdog Daemon Active")
    print(f"Target PID              : {args.pid}")
    print(f"Telemetry Path          : {telem_path}")
    print(f"Hard Limit              : {args.limit_mb:.2f} MiB")
    print(f"Non-PyTorch Allowance   : {args.allowance_mb:.2f} MiB")
    print(f"Effective PyTorch Cap   : {args.limit_mb - args.allowance_mb:.2f} MiB")
    print(f"Polling Interval        : {args.poll_interval:.2f}s")
    print(f"nvidia-smi Available    : {smi_path or 'Not Found'}")
    print("=" * 70)

    # 1. Verify Target PID is valid and alive on startup
    if not is_process_alive(args.pid):
        print(f"[WATCHDOG ERROR] Target PID {args.pid} is not running or invalid.")
        sys.exit(1)

    # 2. Wait for telemetry file to appear
    print(f"[WATCHDOG] Waiting for telemetry file: {telem_path} (timeout: {args.file_wait_timeout}s)...")
    wait_start = time.time()
    while not telem_path.exists():
        if not is_process_alive(args.pid):
            print(f"[WATCHDOG ERROR] Process {args.pid} terminated before creating {telem_path}.")
            sys.exit(2)
        if time.time() - wait_start > args.file_wait_timeout:
            print(f"[WATCHDOG ERROR] Telemetry file {telem_path} was not created within {args.file_wait_timeout}s.")
            sys.exit(3)
        time.sleep(0.5)

    print(f"[WATCHDOG] Telemetry file detected. Beginning continuous surveillance.")

    last_file_pos = 0
    peak_compound_footprint_mb = 0.0
    peak_reserved_mb = 0.0
    peak_allocated_mb = 0.0
    last_iteration_seen = -1
    last_device_poll_time = 0.0

    try:
        while True:
            alive = is_process_alive(args.pid)

            # Optional physical GPU auxiliary monitoring
            now = time.time()
            if smi_path and (now - last_device_poll_time >= 1.0):
                last_device_poll_time = now
                gpu_usage = query_physical_gpu_memory(smi_path)
                if gpu_usage and args.physical_gpu_limit_mb > 0.0:
                    used_phys_mb, total_phys_mb = gpu_usage
                    if used_phys_mb > args.physical_gpu_limit_mb:
                        print(f"\n[WATCHDOG INTERCEPT] System-wide Physical GPU VRAM exceeded ceiling!")
                        print(f"[WATCHDOG INTERCEPT] Used: {used_phys_mb:.2f} MiB > Limit: {args.physical_gpu_limit_mb:.2f} MiB")
                        term_ok, sig_used = terminate_process(args.pid, timeout=args.sigterm_timeout)
                        intercept_data = {
                            "status": "PHYSICAL_GPU_LIMIT_BREACHED",
                            "target_pid": args.pid,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "system_physical_gpu_used_mb": used_phys_mb,
                            "physical_gpu_limit_mb": args.physical_gpu_limit_mb,
                            "termination_signal": sig_used,
                            "graceful": term_ok
                        }
                        with open(output_log, "w", encoding="utf-8") as f:
                            json.dump(intercept_data, f, indent=2)
                        sys.exit(0)

            # Read new telemetry records safely
            if telem_path.exists():
                try:
                    with open(telem_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_file_pos)
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            if not line.endswith("\n"):
                                break
                            
                            line_str = line.strip()
                            if not line_str:
                                last_file_pos = f.tell()
                                continue

                            try:
                                rec = json.loads(line_str)
                            except Exception:
                                last_file_pos = f.tell()
                                continue

                            last_file_pos = f.tell()

                            metrics = extract_memory_from_record(rec, args.allowance_mb)
                            if metrics:
                                iter_num = metrics["iteration"]
                                alloc_mb = metrics["allocated_mb"]
                                res_mb = metrics["reserved_mb"]
                                comp_mb = metrics["compound_footprint_mb"]

                                if iter_num > 0:
                                    last_iteration_seen = iter_num

                                peak_allocated_mb = max(peak_allocated_mb, alloc_mb)
                                peak_reserved_mb = max(peak_reserved_mb, res_mb)
                                peak_compound_footprint_mb = max(peak_compound_footprint_mb, comp_mb)

                                # Evaluate budget breach condition
                                if (comp_mb > args.limit_mb) or (res_mb > args.limit_mb):
                                    print(f"\n" + "!" * 70)
                                    print(f"[WATCHDOG INTERCEPT] Budget breach detected at Iteration {iter_num}!")
                                    print(f"[WATCHDOG INTERCEPT] Compound Footprint: {comp_mb:.2f} MiB (Reserved: {res_mb:.2f} MiB)")
                                    print(f"[WATCHDOG INTERCEPT] Budget Threshold : {args.limit_mb:.2f} MiB")
                                    print("!" * 70)

                                    phys_gpu = query_physical_gpu_memory(smi_path)
                                    term_ok, sig_used = terminate_process(args.pid, timeout=args.sigterm_timeout)

                                    intercept_data = {
                                        "status": "BUDGET_BREACH_INTERCEPTED",
                                        "target_pid": args.pid,
                                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                        "first_observed_iteration": iter_num,
                                        "bounded_iteration_window": [max(0, iter_num - 99), iter_num] if iter_num > 0 else "N/A",
                                        "observed_compound_footprint_mb": comp_mb,
                                        "observed_pytorch_reserved_mb": res_mb,
                                        "observed_pytorch_allocated_mb": alloc_mb,
                                        "hard_limit_mb": args.limit_mb,
                                        "allowance_mb": args.allowance_mb,
                                        "system_physical_gpu_used_mb": phys_gpu[0] if phys_gpu else None,
                                        "system_physical_gpu_total_mb": phys_gpu[1] if phys_gpu else None,
                                        "termination_signal": sig_used,
                                        "graceful_termination": term_ok,
                                        "scientific_disclaimer": (
                                            "BEST-EFFORT EXTERNAL MONITOR. Telemetry written at densification intervals (100 steps). "
                                            "True first budget breach occurred inside bounded window [T_obs - 99, T_obs]."
                                        ),
                                        "violating_telemetry_record": rec
                                    }

                                    output_log.parent.mkdir(parents=True, exist_ok=True)
                                    with open(output_log, "w", encoding="utf-8") as out_f:
                                        json.dump(intercept_data, out_f, indent=2)

                                    print(f"[WATCHDOG] Complete forensic intercept evidence written to: {output_log}")
                                    sys.exit(0)

                except Exception:
                    pass

            if not alive:
                print(f"\n[WATCHDOG] Target process {args.pid} has exited.")
                print(f"[WATCHDOG] Surveillance Summary:")
                print(f"  - Last Iteration Seen       : {last_iteration_seen}")
                print(f"  - Peak PyTorch Allocated    : {peak_allocated_mb:.2f} MiB")
                print(f"  - Peak PyTorch Reserved     : {peak_reserved_mb:.2f} MiB")
                print(f"  - Peak Compound Footprint   : {peak_compound_footprint_mb:.2f} MiB")
                print(f"  - Hard Limit                : {args.limit_mb:.2f} MiB")
                print(f"[WATCHDOG] Verdict: Clean execution within budget. No violations detected.")
                sys.exit(0)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[WATCHDOG] Surveillance manually interrupted via KeyboardInterrupt. Watchdog exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
