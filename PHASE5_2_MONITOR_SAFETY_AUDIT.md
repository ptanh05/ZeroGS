# Phase 5.2 Forensic Safety Audit: External Monitoring & Termination Mechanism

**Document Version:** 1.0  
**Audit Focus:** Safe termination for unconstrained ablation arms (Arms A, B, C) under a 4096 MiB budget on an 8GB physical GPU.  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Execution Environment:** WSL2 Linux (Kernel 6.6, CUDA 12.8, GCC 13)  
**Auditor:** Senior Research Engineer & GPU Systems Auditor  

---

## 1. Executive Summary & Forensic Findings

### Key Audit Conclusions:
1. **Existing Watchdog Status:** **NO EXTERNAL MONITOR CURRENTLY EXISTS IN THE REPOSITORY**.  
   Prior Phase 5 planning referenced an external monitoring daemon conceptually, but no code file or watchdog daemon was implemented.
2. **Telemetry Interval Limitation:**  
   In `train.py:509`, `telemetry_writer.write("mutation_cycle", ...)` is called strictly every `densification_interval` ($100$ iterations) in the unconstrained baseline path.  
   - A memory violation occurring at step $t = 1{,}420$ will **not** be recorded in `telemetry.jsonl` until step $t = 1{,}500$.
   - Therefore, the monitor can identify the **First Observed Violation Iteration ($T_{\text{obs}}$)**, which represents an upper bound: $T_{\text{obs}} \in [T_{\text{true}}, T_{\text{true}} + 100)$.
3. **Distinction between 4GB Budget and 8GB Physical Capacity:**  
   Because the RTX 5060 has $8{,}151\text{ MiB}$ physical VRAM, passing `--disable_zero_oom` allows unconstrained arms to run up to 8GB without CUDA memory allocation failure. An external monitor is strictly required to intercept and terminate the process at 4096 MiB.

---

## 2. Minimal CPU-Side Monitor Specification (`scripts/vram_watchdog.py`)

To guarantee zero source-code modifications to `train.py`, we design an out-of-process, CPU-only watchdog daemon.

### 2.1 Architectural Design & Data Flow
```
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│     train.py Process (PID)      │               │     vram_watchdog.py Process    │
│  - Executes training steps      │               │  - Runs purely on CPU           │
│  - Writes telemetry.jsonl       │               │  - Polls telemetry & OS process │
└────────────────┬────────────────┘               └────────────────┬────────────────┘
                 │ Appends records                                 │ Watches & tails
                 ▼                                                 ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│              experiments/results/ablation_4gb/<arm>/telemetry.jsonl               │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                   If footprint > 4096.0 MiB detected:
                                         │
                                         ▼
                     watchdog sends SIGTERM to train PID
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│          writes <arm>/watchdog_intercept.json (with T_obs, footprint, PID)        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Watchdog Parameter Specification
- **Target File:** Telemetry JSONL path passed via `--telemetry_path`.
- **Target PID:** Process ID of the training process passed via `--pid`.
- **Polling Interval:** $0.25\text{ seconds}$ (tailing file appends).
- **Hard Memory Limit:** $4{,}096.0\text{ MiB}$ ($4{,}294{,}967{,}296\text{ bytes}$).
- **Evaluation Criteria:**
  $$\text{Observed Footprint} = \max(\text{memory\_after.reserved\_mb}, \text{memory\_after.conservative\_footprint\_mb})$$
- **Termination Trigger:** If $\text{Observed Footprint} > 4{,}096.0\text{ MiB}$ on ANY record:
  1. Record timestamp and violating telemetry payload.
  2. Send `os.kill(target_pid, signal.SIGTERM)`.
  3. Wait up to $5.0\text{ seconds}$ for graceful exit (flushing final log buffers).
  4. If process remains alive, send `os.kill(target_pid, signal.SIGKILL)`.
  5. Write `<model_path>/watchdog_intercept.json`.

---

## 3. Implementation of the Standalone Watchdog

The complete CPU-only monitor script specification to be placed in `scripts/vram_watchdog.py`:

```python
#!/usr/bin/env python3
"""
ZeroGS CPU-Side VRAM Watchdog Daemon.
Monitors telemetry.jsonl for unconstrained arms (Arms A, B, C) and terminates
training immediately upon observing a breach of the 4096.0 MiB hard VRAM budget.
"""

import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="ZeroGS VRAM Watchdog")
    parser.add_argument("--pid", type=int, required=True, help="PID of training process")
    parser.add_argument("--telemetry_path", type=str, required=True, help="Path to telemetry.jsonl")
    parser.add_argument("--limit_mb", type=float, default=4096.0, help="Hard VRAM budget in MiB")
    parser.add_argument("--poll_interval", type=float, default=0.25, help="Polling interval in seconds")
    parser.add_argument("--output_log", type=str, default=None, help="Path to write watchdog_intercept.json")
    return parser.parse_args()


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate_process(pid: int, timeout: float = 5.0) -> bool:
    if not is_process_alive(pid):
        return True
    print(f"[WATCHDOG] Sending SIGTERM to PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True

    start = time.time()
    while time.time() - start < timeout:
        if not is_process_alive(pid):
            print(f"[WATCHDOG] PID {pid} terminated gracefully.")
            return True
        time.sleep(0.1)

    print(f"[WATCHDOG] PID {pid} did not exit within {timeout}s. Sending SIGKILL...")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return not is_process_alive(pid)


def main():
    args = parse_args()
    telem_file = Path(args.telemetry_path)
    output_log = Path(args.output_log) if args.output_log else telem_file.parent / "watchdog_intercept.json"

    print(f"[WATCHDOG] Active. Monitoring PID {args.pid} via {telem_file}")
    print(f"[WATCHDOG] Budget Threshold: {args.limit_mb:.2f} MiB")

    last_pos = 0
    while is_process_alive(args.pid):
        if telem_file.exists():
            with open(telem_file, "r", encoding="utf-8") as f:
                f.seek(last_pos)
                lines = f.readlines()
                last_pos = f.tell()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                # Check all memory fields
                iteration = rec.get("iteration", -1)
                footprint_mb = 0.0

                if "memory_after" in rec:
                    m = rec["memory_after"]
                    footprint_mb = max(m.get("reserved_mb", 0.0), m.get("conservative_footprint_mb", 0.0))
                elif "memory" in rec:
                    m = rec["memory"]
                    footprint_mb = max(m.get("reserved_mb", 0.0), m.get("conservative_footprint_mb", 0.0))

                if footprint_mb > args.limit_mb:
                    print(f"\n[WATCHDOG INTERCEPT] Budget breach detected at Iteration {iteration}!")
                    print(f"[WATCHDOG INTERCEPT] Observed Footprint: {footprint_mb:.2f} MiB > Limit: {args.limit_mb:.2f} MiB")
                    
                    # Terminate training
                    terminate_process(args.pid)

                    # Write intercept record
                    intercept_data = {
                        "event": "watchdog_budget_breach",
                        "target_pid": args.pid,
                        "first_observed_iteration": iteration,
                        "observed_footprint_mb": footprint_mb,
                        "hard_limit_mb": args.limit_mb,
                        "timestamp": time.time(),
                        "telemetry_record": rec
                    }
                    with open(output_log, "w", encoding="utf-8") as out_f:
                        json.dump(intercept_data, out_f, indent=2)
                    print(f"[WATCHDOG] Intercept audit logged to {output_log}")
                    sys.exit(0)

        time.sleep(args.poll_interval)

    print(f"[WATCHDOG] Target process {args.pid} exited cleanly without violating {args.limit_mb:.2f} MiB.")


if __name__ == "__main__":
    main()
```

---

## 4. CPU-Only Synthetic Test Plan

To verify the watchdog daemon prior to any GPU training, run the following deterministic synthetic test:

### Test Protocol:
1. Launch a dummy Python child process that writes simulated `telemetry.jsonl` entries every 100 iterations.
2. At iteration 600, write a record with `reserved_mb = 3500.0`.
3. At iteration 700, write a record with `reserved_mb = 4250.0` ($> 4096.0\text{ MiB}$).
4. Verify that `vram_watchdog.py`:
   - Detects iteration 700 within $< 0.3\text{ seconds}$.
   - Sends `SIGTERM` and confirms the dummy process terminates.
   - Writes `watchdog_intercept.json` with exact payload.
   - Exits with return code 0.

---

## 5. First Observed ($T_{\text{obs}}$) vs. True First Violation ($T_{\text{true}}$)

Because `train.py` densification cycles occur at intervals of $\Delta t = 100$:
$$T_{\text{true}} \le T_{\text{obs}} < T_{\text{true}} + 100$$
- In the publication manuscript, we report:
  $$\text{Budget Failure Milestone: } T_{\text{obs}} \quad (\text{Bounded window: } [T_{\text{obs}}-99, T_{\text{obs}}])$$
- This scientific distinction prevents reviewers from claiming that the paper overstates temporal precision.
