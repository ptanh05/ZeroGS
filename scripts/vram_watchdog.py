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
import uuid
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


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    _kernel32 = ctypes.windll.kernel32
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259

    def _is_process_alive_win32(pid: int) -> bool:
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if _kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == _STILL_ACTIVE
            return False
        finally:
            _kernel32.CloseHandle(handle)


def is_process_alive(pid: int) -> bool:
    """
    Check if process with given PID exists, is accessible, and is actively running.
    Handles Unix zombie/defunct state where process has terminated but not yet reaped,
    and handles Windows process handle state via Win32 GetExitCodeProcess.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        return _is_process_alive_win32(pid)

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


def write_json_atomic(target_path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload to a temporary file in the same directory, then os.replace() atomically."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f"{target_path.name}.tmp_{uuid.uuid4().hex[:8]}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def verify_telemetry_flushed(telem_path: Path) -> Tuple[bool, str]:
    """Verify that telemetry JSONL exists, is non-empty, and ends with a complete valid JSON record."""
    if not telem_path.exists():
        return False, "TELEMETRY_NOT_FOUND"
    try:
        with open(telem_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return False, "TELEMETRY_EMPTY"
            # Seek backwards in chunks so lines longer than 4096 bytes are never truncated.
            chunk_size = 8192
            pos = size
            trailing_bytes = b""
            while pos > 0:
                step = min(chunk_size, pos)
                pos -= step
                f.seek(pos)
                trailing_bytes = f.read(step) + trailing_bytes
                stripped = trailing_bytes.rstrip(b"\r\n\t ")
                if stripped and b"\n" in stripped:
                    break
            text = trailing_bytes.decode("utf-8", errors="replace")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if not lines:
                return False, "TELEMETRY_NO_LINES"
            last_line = lines[-1]
            json.loads(last_line)
            return True, "TELEMETRY_FLUSHED_VALID"
    except Exception as e:
        return False, f"TELEMETRY_CORRUPT: {e}"


def verify_checkpoints_integrity(directory: Path) -> Dict[str, Any]:
    """
    Scan for chkpnt*.pth files in directory and verify each is a valid non-empty archive.
    Explicitly tracks and isolates orphaned temporary files (.tmp*), ensuring they are
    never misidentified as valid checkpoints.
    """
    import zipfile
    chkpnts = sorted(directory.glob("chkpnt*.pth"))
    orphaned_tmp = sorted([p.name for p in directory.glob("chkpnt*.tmp*")])
    results = []
    all_valid = True
    for cp in chkpnts:
        size = cp.stat().st_size
        is_valid = False
        reason = "OK"
        if size == 0:
            reason = "ZERO_BYTES"
            all_valid = False
        else:
            try:
                if zipfile.is_zipfile(str(cp)):
                    with zipfile.ZipFile(str(cp), "r") as zf:
                        bad_file = zf.testzip()
                        if bad_file is None:
                            is_valid = True
                        else:
                            reason = f"CORRUPT_CRC: {bad_file}"
                            all_valid = False
                elif size > 1024:
                    is_valid = True
                else:
                    reason = "INVALID_SIZE"
                    all_valid = False
            except Exception as e:
                reason = f"VERIFY_ERROR: {e}"
                all_valid = False

        results.append({
            "path": str(cp),
            "filename": cp.name,
            "size_bytes": size,
            "valid": is_valid,
            "status": reason
        })

    return {
        "count": len(chkpnts),
        "all_valid": all_valid,
        "orphaned_tmp_count": len(orphaned_tmp),
        "orphaned_tmp_files": orphaned_tmp,
        "checkpoints": results
    }


def terminate_process(pid: int, timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Terminate target PID.
    On Windows: Uses TerminateProcess (via os.kill(pid, signal.SIGTERM)).
    This is an abrupt, uncooperative hard-kill without Python signal handlers or cleanup.
    Returns (graceful=False, signal_used="TerminateProcess").

    On POSIX/Linux: Sends SIGTERM, waits up to timeout seconds, then escalates to SIGKILL.
    Returns (graceful=True, signal_used="SIGTERM") or (graceful=False, signal_used="SIGKILL").
    """
    if not is_process_alive(pid):
        return False, "ALREADY_TERMINATED"

    if sys.platform == "win32":
        print(f"[WATCHDOG] Windows detected: Invoking TerminateProcess on PID {pid} (uncooperative shutdown)...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            print(f"[WATCHDOG] TerminateProcess error: {e}")
            return False, "TERMINATE_FAILED"

        time.sleep(0.1)
        alive = is_process_alive(pid)
        if not alive:
            print(f"[WATCHDOG] PID {pid} terminated via TerminateProcess. WARNING: Abrupt termination, not graceful.")
            return False, "TerminateProcess"
        else:
            print(f"[WATCHDOG ERROR] PID {pid} still alive after TerminateProcess!")
            return False, "TERMINATE_FAILED"

    print(f"[WATCHDOG] Sending SIGTERM to PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"[WATCHDOG] Failed to send SIGTERM: {e}")
        return False, "SIGTERM_FAILED"

    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_process_alive(pid):
            print(f"[WATCHDOG] PID {pid} terminated gracefully via SIGTERM within {time.time() - start_time:.2f}s.")
            return True, "SIGTERM"
        time.sleep(0.1)

    print(f"[WATCHDOG] PID {pid} did not exit within {timeout}s grace period. Escalating to SIGKILL...")
    try:
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, sigkill)
    except OSError as e:
        print(f"[WATCHDOG] Error sending SIGKILL: {e}")

    time.sleep(0.5)
    alive = is_process_alive(pid)
    if not alive:
        print(f"[WATCHDOG] PID {pid} successfully terminated via SIGKILL.")
        return False, "SIGKILL"
    else:
        print(f"[WATCHDOG ERROR] PID {pid} could not be terminated!")
        return False, "SIGKILL_FAILED"


def execute_cooperative_or_forced_shutdown(
    pid: int,
    stop_flag: Path,
    stop_payload: Dict[str, Any],
    telem_path: Path,
    cooperative_timeout: float,
    sigterm_timeout: float,
) -> Dict[str, Any]:
    """
    Request cooperative shutdown atomically via stop_flag, wait up to cooperative_timeout,
    and fall back to forced termination if still running. Collects post-exit forensic evidence.
    """
    try:
        write_json_atomic(stop_flag, stop_payload)
        print(f"[WATCHDOG] Cooperative stop requested via: {stop_flag}")
    except Exception as e:
        print(f"[WATCHDOG] Warning: Failed to write stop.flag: {e}")

    print(f"[WATCHDOG] Waiting up to {cooperative_timeout:.1f}s for cooperative shutdown...")
    coop_start = time.time()
    coop_success = False
    while time.time() - coop_start < cooperative_timeout:
        if not is_process_alive(pid):
            coop_success = True
            print(f"[WATCHDOG] Process {pid} shut down cooperatively within {time.time() - coop_start:.2f}s.")
            break
        time.sleep(0.1)

    if coop_success:
        term_ok = True
        sig_used = "COOPERATIVE_STOP"
    else:
        print(f"[WATCHDOG] Process did not shut down within {cooperative_timeout:.1f}s grace period. Escalating to forced termination...")
        term_ok, sig_used = terminate_process(pid, timeout=sigterm_timeout)

    telem_flushed, telem_status = verify_telemetry_flushed(telem_path)
    cp_report = verify_checkpoints_integrity(telem_path.parent)

    manifest_file = telem_path.parent / "cooperative_shutdown.json"
    coop_manifest = None
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as mf:
                coop_manifest = json.load(mf)
        except Exception:
            pass

    graceful_termination = bool(coop_success) or (term_ok and sys.platform != "win32")
    windows_limitation = (
        "Cooperative stop succeeded gracefully; emergency checkpoint verified."
        if coop_success else (
            "On Windows, TerminateProcess was invoked abruptly after timeout. "
            "Only pre-existing checkpoints flushed prior to intercept remain intact."
            if sys.platform == "win32" else "N/A (POSIX SIGTERM used)"
        )
    )

    return {
        "coop_success": coop_success,
        "termination_signal": sig_used,
        "graceful_termination": graceful_termination,
        "cooperative_shutdown_manifest": coop_manifest,
        "telemetry_flushed": telem_flushed,
        "telemetry_flush_status": telem_status,
        "checkpoints_verified": cp_report,
        "windows_safety_limitation": windows_limitation,
    }


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
    parser.add_argument("--cooperative_timeout", type=float, default=5.0,
                        help="Seconds to wait for process to shut down cooperatively via stop.flag before force termination")
    parser.add_argument("--stop_file", type=str, default=None,
                        help="Explicit path to stop flag (defaults to <telemetry_dir>/stop.flag)")
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

    # Resolve stop_flag path
    stop_flag = Path(args.stop_file) if args.stop_file else (telem_path.parent / "stop.flag")

    # Clean stale stop.flag targeting a prior different PID
    if stop_flag.exists():
        try:
            with open(stop_flag, "r", encoding="utf-8") as sf:
                old_data = json.loads(sf.read())
            old_pid = old_data.get("target_pid")
            if old_pid is not None and old_pid != args.pid:
                stop_flag.unlink(missing_ok=True)
                print(f"[WATCHDOG] Removed stale stop.flag targeting prior PID {old_pid}")
        except Exception:
            pass

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
                        stop_payload = {
                            "reason": "PHYSICAL_GPU_LIMIT_BREACHED",
                            "target_pid": args.pid,
                            "system_physical_gpu_used_mb": used_phys_mb,
                            "physical_gpu_limit_mb": args.physical_gpu_limit_mb,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()
                        }
                        outcome = execute_cooperative_or_forced_shutdown(
                            pid=args.pid,
                            stop_flag=stop_flag,
                            stop_payload=stop_payload,
                            telem_path=telem_path,
                            cooperative_timeout=args.cooperative_timeout,
                            sigterm_timeout=args.sigterm_timeout,
                        )
                        intercept_data = {
                            "status": "PHYSICAL_GPU_LIMIT_BREACHED",
                            "target_pid": args.pid,
                            "platform": sys.platform,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "system_physical_gpu_used_mb": used_phys_mb,
                            "system_physical_gpu_total_mb": total_phys_mb,
                            "physical_gpu_limit_mb": args.physical_gpu_limit_mb,
                            "termination_signal": outcome["termination_signal"],
                            "graceful": outcome["graceful_termination"],
                            "graceful_termination": outcome["graceful_termination"],
                            "cooperative_shutdown_manifest": outcome["cooperative_shutdown_manifest"],
                            "telemetry_flushed": outcome["telemetry_flushed"],
                            "telemetry_flush_status": outcome["telemetry_flush_status"],
                            "checkpoints_verified": outcome["checkpoints_verified"],
                            "windows_safety_limitation": outcome["windows_safety_limitation"],
                        }
                        write_json_atomic(output_log, intercept_data)
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

                                    stop_payload = {
                                        "reason": "VRAM_LIMIT_EXCEEDED",
                                        "target_pid": args.pid,
                                        "observed_compound_footprint_mb": comp_mb,
                                        "observed_pytorch_reserved_mb": res_mb,
                                        "observed_pytorch_allocated_mb": alloc_mb,
                                        "hard_limit_mb": args.limit_mb,
                                        "iteration": iter_num,
                                        "timestamp_utc": datetime.now(timezone.utc).isoformat()
                                    }
                                    outcome = execute_cooperative_or_forced_shutdown(
                                        pid=args.pid,
                                        stop_flag=stop_flag,
                                        stop_payload=stop_payload,
                                        telem_path=telem_path,
                                        cooperative_timeout=args.cooperative_timeout,
                                        sigterm_timeout=args.sigterm_timeout,
                                    )

                                    intercept_data = {
                                        "status": "BUDGET_BREACH_INTERCEPTED",
                                        "target_pid": args.pid,
                                        "platform": sys.platform,
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
                                        "termination_signal": outcome["termination_signal"],
                                        "graceful_termination": outcome["graceful_termination"],
                                        "cooperative_shutdown_manifest": outcome["cooperative_shutdown_manifest"],
                                        "telemetry_flushed": outcome["telemetry_flushed"],
                                        "telemetry_flush_status": outcome["telemetry_flush_status"],
                                        "checkpoints_verified": outcome["checkpoints_verified"],
                                        "windows_safety_limitation": outcome["windows_safety_limitation"],
                                        "scientific_disclaimer": (
                                            "BEST-EFFORT EXTERNAL MONITOR. Telemetry written at densification intervals (100 steps). "
                                            "True first budget breach occurred inside bounded window [T_obs - 99, T_obs]."
                                        ),
                                        "violating_telemetry_record": rec
                                    }

                                    write_json_atomic(output_log, intercept_data)
                                    print(f"[WATCHDOG] Complete forensic intercept evidence written to: {output_log}")
                                    sys.exit(0)

                except Exception:
                    pass

            if not alive:
                telem_flushed, telem_status = verify_telemetry_flushed(telem_path)
                cp_report = verify_checkpoints_integrity(telem_path.parent)
                print(f"\n[WATCHDOG] Target process {args.pid} has exited.")
                print(f"[WATCHDOG] Surveillance Summary:")
                print(f"  - Last Iteration Seen       : {last_iteration_seen}")
                print(f"  - Peak PyTorch Allocated    : {peak_allocated_mb:.2f} MiB")
                print(f"  - Peak PyTorch Reserved     : {peak_reserved_mb:.2f} MiB")
                print(f"  - Peak Compound Footprint   : {peak_compound_footprint_mb:.2f} MiB")
                print(f"  - Hard Limit                : {args.limit_mb:.2f} MiB")
                print(f"  - Telemetry Flushed         : {telem_flushed} ({telem_status})")
                print(f"  - Checkpoints Verified      : {cp_report['count']} files (all valid: {cp_report['all_valid']})")
                print(f"[WATCHDOG] Verdict: Clean execution within budget. No violations detected.")
                sys.exit(0)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[WATCHDOG] Surveillance manually interrupted via KeyboardInterrupt. Watchdog exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
