# Forensic Audit: Extended Grace Smoke Test Anomalies (PV-SRA-3DGS V2)

**Project:** Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3D Gaussian Splatting (PV-SRA-3DGS)  
**Date:** 2026-11-04  
**Audit Scope:** 4,000-Iteration Smoke Test (`output/smoke_test_extended_4k`, `EXTENDED_GRACE_SMOKE_REPORT.md`)  
**Methodology:** Read-only static code inspection, telemetry log parsing, checkpoint parameter analysis, and mathematical reconciliation.  
**Verdict:** **PASS WITH LIMITATIONS**

---

## 1. VRAM Measurement Forensic Audit

### 1.1 The Anomaly
In `EXTENDED_GRACE_SMOKE_REPORT.md`, the reported memory values were:
- **Peak Allocated:** 373.0 MB
- **Peak Reserved:** 500.0 MB

In contrast, earlier 1,000-step smoke tests (`smoke_test_bonsai_A3_retry2`) and full 30K benchmark runs reported:
- **Active / Peak Allocated:** ~2.0 – 2.5 GB
- **Peak Reserved:** ~4.9 – 6.2 GB

### 1.2 Root Cause Analysis

Our audit identified two distinct technical factors that caused this discrepancy:

#### Factor A: Dual Downsampling via CLI Arguments
The command executed for the extended smoke test was:
```bash
python train.py -s data/bonsai -i images_4 ... -r 4 --quiet
```
In `scene/dataset_readers.py` and `utils/camera_utils.py`:
- Passing `-i images_4` instructs COLMAP camera loading to read pre-downsampled images from `data/bonsai/images_4/` (resolution: $780 \times 520$, downscaled $4\times$ from original $3118 \times 2078$).
- In `utils/camera_utils.py` (lines 43–44):
  ```python
  if args.resolution in [1, 2, 4, 8]:
      resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
  ```
  Passing `-r 4` instructed `loadCam()` to divide the input image dimensions by an *additional* factor of 4.
- As a result, the loaded camera resolution on CUDA was:
  $$\text{Resolution} = \left(\frac{780}{4}, \frac{520}{4}\right) = \mathbf{(195, 130)}$$
- In canonical 3DGS, all 292 dataset cameras are pinned in GPU memory (`self.original_image.to("cuda")`). At canonical $780 \times 520$ resolution, 292 float32 RGB + alpha images occupy **1,807.2 MB** of VRAM. At $195 \times 130$ resolution, they occupy only **112.9 MB** (a 16-fold reduction).
- Furthermore, rasterization framebuffers and tile sorting buffers scaled down proportionally from $780 \times 520$ to $195 \times 130$, reducing activation memory from ~500 MB down to ~35 MB.

#### Factor B: Peak Reset Scope in Telemetry Collection
In `train.py` (lines 434–437):
```python
if iteration % opt.densification_interval == 0:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
```
`torch.cuda.reset_peak_memory_stats()` was called at every densification interval (every 100 iterations), including at **iteration 4,000**.
At the end of training (lines 898–902):
```python
telemetry_writer.write(
    "training_end",
    final_gaussians=int(gaussians.get_xyz.shape[0]),
    admission=admission_ctrl.get_telemetry(),
    memory=capture_cuda_memory().to_dict(),
)
```
The snapshot recorded at `training_end` only reflected the peak memory of **iteration 4,000 itself** (the single step after the reset), which was 372.979 MB allocated and 500.0 MB reserved.

### 1.3 True Global Memory Breakdown Across the 4,000 Iterations
Parsing all 35 mutation cycles across the entire run reveals the true global extrema:

| Measurement Dimension | Reported in Initial Summary | True Global Value Across 4K Run | Observation Point |
|---|---|---|---|
| **1. PyTorch Peak Allocated** | 373.0 MB | **575.82 MB** | Iteration 3,000 (287,397 primitives) |
| **2. PyTorch Peak Reserved** | 500.0 MB | **962.00 MB** | Iteration 3,000 |
| **3. Peak Device VRAM** | Not reported | **Unmonitored** | Standalone `train.py` does not poll `nvidia-smi` |
| **4. Controller Estimated Footprint** | 498.0 MB | **962.00 MB** | Iteration 3,000 (`max(allocated, reserved)`) |
| **5. Minimum Available Headroom** | 4,968.0 MB | **4,748.00 MB** | Iteration 3,100 (Hard limit 6,144 MB) |

> [!IMPORTANT]
> The low memory footprint observed in the 4K smoke test is an artifact of dual downsampling (`-i images_4` with `-r 4` yielding $195 \times 130$ images). In full benchmark runs, `run_single_benchmark.py` passes `-i images_4` without `-r 4` (yielding $780 \times 520$), where baseline image VRAM is ~1.8 GB and total allocated VRAM is ~2.4–3.9 GB. Device-level memory headroom must NOT be derived from PyTorch reserved memory alone.

---

## 2. Geometry Pruning Audit — Iteration 3,100

### 2.1 The Observation
At iteration 3,100 (the first mutation cycle inside the reset cooldown), **76,521 Gaussians** were removed, despite opacity-based pruning being suppressed (`n_protected_by_cooldown = 89,259`).
At iteration 3,400 (cooldown expiration), **70,859 Gaussians** were removed.

### 2.2 Inspection of Pruning Logic and Coordinate Systems
In `train.py` (lines 649, 691–698):
```python
size_threshold = 20 if iteration > opt.opacity_reset_interval else None
...
hygiene_mask = dead_opacity_mask
if size_threshold is not None:
    if hasattr(gaussians, "max_radii2D"):
        hygiene_mask = torch.logical_or(hygiene_mask, gaussians.max_radii2D > size_threshold)
    max_world_size_ratio = getattr(opt, "max_world_size_ratio", 0.1)
    big_ws = torch.max(gaussians.get_scaling, dim=1).values > (max_world_size_ratio * scene_extent)
    hygiene_mask = torch.logical_or(hygiene_mask, big_ws)
```

1. **Boolean Combination:**
   The eviction mask is a logical **OR**:
   $$\mathcal{M}_{\text{hygiene}} = \mathcal{M}_{\text{opacity}} \lor \mathcal{M}_{\text{radii\_2D}} \lor \mathcal{M}_{\text{world\_scale}}$$
   During cooldown (iteration 3,100), $\mathcal{M}_{\text{opacity}} = \mathbf{0}$, reducing the eviction mask strictly to geometry:
   $$\mathcal{M}_{\text{hygiene}} = (\mathbf{r}_{2D} > 20) \lor (\mathbf{s}_{\text{world}} > 0.1 \cdot S_{\text{scene}})$$

2. **Scheduling Defect / Latency:**
   Notice that `big_ws` (world-space scale filter) is nested *inside* `if size_threshold is not None:`.
   - `size_threshold` is `None` for all iterations $t \le 3,000$.
   - Consequently, for the first 30 mutation cycles (iterations 600 to 3,000), **neither screen-space radii nor world-space scale was pruned**.
   - Primitives expanded freely in scale for 3,000 iterations without any size regularization.

3. **Quantitative Decomposition at Step 3,000:**
   Inspection of `output/smoke_test_extended_4k/chkpnt3000.pth` (287,397 Gaussians, $S_{\text{scene}} = 5.1656$, threshold $= 0.5166$):
   - Gaussians with $\max \mathbf{s}_{\text{world}} > 0.5166$: **34,228** primitives.
   - Gaussians with $\mathbf{r}_{2D} > 20$: Accumulated over steps 3,001–3,100.
   - Total unique Gaussians meeting the union condition at step 3,100: **76,521**.
   - Gaussian count before pruning: **287,397**.
   - Gaussian count after pruning and densification: **213,635** (net removal: 73,762).

4. **Trajectory Following Iteration 3,100:**
   - Iteration 3,200: 6,496 removed.
   - Iteration 3,300: 3,157 removed.
   The rapid decay (76.5K $\to$ 6.5K $\to$ 3.2K) proves this was a one-time purge of floaters that had accumulated during the unconstrained 1–3,000 step window.

5. **The Iteration 3,400 Purge (70,859 Removals):**
   - At iteration 3,000, all opacities were reset to 0.01.
   - During steps 3,100, 3,200, and 3,300, opacity pruning was suspended by $\tau_{\text{reset}} = 400$.
   - At iteration 3,400, cooldown expired. Exactly 70,859 primitives that received minimal view supervision or lay in background regions failed to cross $\alpha \ge 0.005$ and were pruned in a single batch.
   - Newborn primitives were protected by $\tau_{\text{age}}$ (1,973 young Gaussians shielded at step 3,400).
   - Following step 3,400, pruning stabilized to normal levels (5.9K at step 3500, down to 1.5K at step 4000).

---

## 3. Split Quota Semantics Forensic Audit

### 3.1 Mathematical Semantics in `joint_knapsack_allocate()`
Location: `zerogs/marginal_allocator.py` (lines 171–257)

1. **What $K_{\max}$ Represents:**
   ```python
   capacity_ceiling = total_candidates
   if max_items is not None:
       capacity_ceiling = min(capacity_ceiling, max(0, int(max_items)))
   ```
   $K_{\max}$ (`capacity_ceiling`) is the upper bound on total admitted mutation actions:
   $$K_{\max} = \min\left(N_{\text{clone\_cand}} + N_{\text{split\_cand}}, M_{\text{iter\_cap}}, M_{\text{pop\_cap}}\right)$$
   where $M_{\text{iter\_cap}} = 5,000$ and $M_{\text{pop\_cap}} = \max(0, 1.5\times 10^6 - N_{\text{current}})$.

2. **Unit of Quota Measurement:**
   - Quota is measured in **admitted split actions** (number of parent primitives selected for splitting), NOT candidate count, generated children count, or byte cost.
   - Target split quota:
     $$K_{\text{split\_target}} = \min\left(N_{\text{split\_cand}}, \left\lceil \kappa_{\text{split}} \times K_{\max} \right\rceil\right)$$

3. **Primitive Generation & Parent Handling:**
   In `densify_and_split()` (`scene/gaussian_model.py`, lines 606–630):
   - A split action selects 1 parent primitive and generates $N = 2$ child primitives.
   - `prune_points(selected_pts_mask)` is called on the parent primitive immediately following child generation.
   - Net primitive increase per split action is $+1$ ($2 \text{ children} - 1 \text{ parent} = +1$).
   - Gross primitives created per split action is 2.

4. **Feasibility Filtering Before Quota Enforcement:**
   In `zerogs/marginal_allocator.py` (lines 191–201):
   ```python
   actual_cost = cost_split + operation_overhead
   if actual_cost <= remaining_bytes:
       remaining_bytes -= actual_cost
       final_split_mask[s_idx] = True
       ...
   else:
       break
   ```
   Every candidate in Phase 1 is strictly evaluated against remaining transient mutation budget `remaining_bytes`. If a split would cause budget violation, the loop terminates immediately (`break`). Infeasible candidates are **never** forced into admission.

5. **Budget Boundary Invariant:**
   The split quota can **never** exceed the available VRAM budget. Budget safety strictly dominates quota satisfaction.

### 3.2 Empirical Verification of Split Share from Raw Telemetry
Every mutation cycle was recalculated directly from raw action counters in `training_telemetry.jsonl`:
$$\text{Recalculated Share} = \frac{N_{\text{split\_executed}}}{N_{\text{clone\_executed}} + N_{\text{split\_executed}}}$$

- **Match Rate:** 35 / 35 cycles match logged `actual_split_share` with error $< 10^{-6}$.
- **Unconstrained Phase (Steps 600–2000):** Split share ranged from **0.4582 to 0.9652** (demand-driven, exceeding 40%).
- **Saturation Phase (Steps 2100–3000):** Candidate demand exceeded $M_{\text{iter\_cap}} = 5,000$. The allocator admitted exactly 3,000 clones and 2,000 splits:
  $$\text{Split Share} = \frac{2,000}{3,000 + 2,000} = \mathbf{0.4000}$$
  The 40% floor was satisfied across 100% of saturation cycles.
- **Post-Cooldown Phase (Steps 3100–4000):** Split share ranged from **0.4370 to 0.7546**.

---

## 4. Quality Impact Assessment

- **Evaluation Outputs:** Checkpoints `chkpnt3000.pth` and `chkpnt4000.pth` do not possess photometric evaluation files (`results.json`, `eval/`).
- **Resolution Consideration:** Because training was executed at $195 \times 130$ (dual downsampled), photometric metrics would not reflect standard benchmark performance.
- **Explicit Statement:**
  > [!NOTE]
  > The 4,000-iteration extended smoke test validates **lifecycle mechanics, state machine transitions, cooldown timing, tensor alignment, and memory boundary adherence only**. It does NOT validate photometric reconstruction quality (PSNR, SSIM, LPIPS). Reconstruction quality will be rigorously measured on the 37 canonical holdout test cameras during the 30,000-iteration benchmark runs.

---

## 5. Audit Findings & Resolution Matrix

| Audit Area | Finding | Severity | Resolution / Clarification |
|---|---|---|---|
| **VRAM Measurement** | Peak allocated (373 MB) and reserved (500 MB) were artificially low due to dual downsampling (`-i images_4 -r 4`) and peak reset at step 4,000. | Medium (Telemetry) | True global peak across 4K run was **576 MB allocated** and **962 MB reserved**. 30K benchmark protocol uses `-i images_4` without `-r 4` ($780 \times 520$), correctly loading full baseline VRAM (~2.4–3.9 GB). |
| **Geometry Pruning (3,100)** | 76,521 floaters removed because world-scale pruning was deferred until step 3,000 when `size_threshold` activated. | Low (Canonical) | Matches canonical 3DGS behavior where floater pruning is deferred to the first reset. Rapid decay in subsequent steps proves stability. |
| **Opacity Pruning (3,400)** | 70,859 dead-opacity Gaussians purged after cooldown expired. | Expected | Direct consequence of $\tau_{\text{reset}} = 400$ deferral; Gaussians that failed to recover opacity were purged in batch. |
| **Split Quota Semantics** | Quota strictly measured by admitted actions; feasibility checked; budget never breached; 100% telemetry consistency. | Verified | Mathematical formulation is sound and verified across all 35 cycles. |
| **Quality Validation** | No eval metrics exist for 4K test. | Informational | Explicitly scoped to lifecycle mechanics. Full 30K benchmark will measure 7K and 30K metrics. |

---

## 6. Final Verdict

$$\mathbf{VERDICT:}\quad \text{\textbf{PASS WITH LIMITATIONS}}$$

### Limitations Acknowledged:
1. Memory in the 4K run was measured under dual downsampling ($195 \times 130$ pixels), reducing baseline dataset image VRAM from ~1.8 GB to ~113 MB. Benchmark runs must use canonical resolution ($780 \times 520$).
2. Device-level hardware VRAM was not polled via `nvidia-smi` during standalone `train.py` execution; hardware telemetry must be captured via `run_single_benchmark.py`'s `DeviceMemoryMonitor`.
3. The 4K test validates lifecycle mechanics only; photometric reconstruction quality will be confirmed in the full 30,000-iteration benchmark.

