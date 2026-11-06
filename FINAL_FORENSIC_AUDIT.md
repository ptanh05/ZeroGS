# ZeroGS Final Forensic Research Audit
## Comprehensive Independent Audit of Code, Physical Artifacts, Telemetry, and Manuscript Claims
### Project: PV-SRA-3DGS (Peak-VRAM-Constrained Spatial Resource Allocation for Memory-Bounded 3DGS)

**Audit Completion Date:** 2026-11-06  
**Auditor:** Senior Research Scientist, ML Engineer & Independent Research Auditor  
**Hardware Environment:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MiB Physical VRAM, Windows 11 WDDM)  
**Execution Protocol:** 100% Serial Physical Hardware Runs (Zero Concurrency, Zero Synthetic Data)  

---

## 1. Executive Summary

This independent forensic audit was conducted on the entire ZeroGS repository, physical output directories, source code, test suites, intermediate reports, and manuscript drafts. The audit was performed under strict evidentiary rules: **raw disk artifacts (checkpoints, PLY point clouds, raw telemetry logs, and evaluation JSONs) were treated as the sole source of truth**.

### Summary of Audit Verdicts
- **Physical Runs Verified:** Exactly **23 non-interrupted physical benchmark runs** (30,000 steps each) were validated directly on disk.
- **Primary Scientific Discovery Confirmed:** The **Over-Pruning Dilemma** and the breakthrough performance of the **Preservation-Centric Dynamic Inflow architecture (`ABL-HP`)** are **100% physically verified and reproduced**:
  - `bonsai` (Indoor Bounded): **$31.8305 \pm 0.4789\text{ dB}$** vs Baseline V4 $20.7455 \pm 1.8690\text{ dB}$ (**$+11.0850\text{ dB}$ paired gain**, $p = 0.01451$, Cohen's $d = 4.7406$).
  - `garden` (Outdoor Unbounded): **$24.9277\text{ dB}$** vs Baseline V4 $18.6656\text{ dB}$ (**$+6.2621\text{ dB}$ paired gain**, $+93.8\%$ SSIM increase).
  - `flowers` (Complex Flora): **$20.2994\text{ dB}$** vs Baseline V4 $17.3572\text{ dB}$ (**$+2.9422\text{ dB}$ paired gain**, $+53.1\%$ SSIM increase).
- **Physical Memory Safety Verified:** Across all 23 physical benchmark runs on an 8GB GPU, **zero Out-Of-Memory (OOM) events** occurred, and peak device memory remained strictly bounded within **$6,906.0$ to $7,656.0\text{ MiB}$**.
- **Critical Discrepancies Uncovered:**
  1. *Candidate Conservation Boundary:* While $\text{Requested} = \text{Admissible Cap} + \text{Rejected by Cap}$ holds algebraically at the admission interface, a **telemetry timing defect** logged the pre-prune decision rather than the post-prune reallocation on `PRUNE_FIRST` cycles, causing an apparent execution delta in 8 cycles on `flowers` and 5 cycles on `bonsai`.
  2. *Memory Constant Conflation:* $1{,}137\text{ B}$ is the **transient cloning cost** ($944\text{ B} \times 1.20458$), NOT the base persistent Gaussian footprint ($944\text{ B}$).
  3. *Unconstrained Reference Scope:* $V_{\text{ref}}$ was executed only on `bonsai` ($29.59\text{ dB}$). Claiming that `ABL-HP` outperforms $V_{\text{ref}}$ on `garden` and `flowers` is currently **UNSUPPORTED** because physical $V_{\text{ref}}$ runs were not performed on those scenes.
  4. *Holdout View Arithmetic Errors:* `paper/EXPERIMENTAL_RESULTS.md` misstated the holdout views for `garden` as 23 (true value: 24) and `flowers` as 21 (true value: 22).

---

## 2. Forensic Audit of Physical Benchmark Runs

### 2.1 Master Experiment Ledger Overview
All 23 physical runs have been cataloged in [`MASTER_EXPERIMENT_LEDGER.csv`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/MASTER_EXPERIMENT_LEDGER.csv). Every single entry was verified for:
1. `chkpnt30000.pth` existence and readability via `torch.load(..., weights_only=False)`.
2. `point_cloud.ply` existence and vertex count matching telemetry.
3. `results.json` at iteration 30,000 containing full PSNR, SSIM, and LPIPS metrics.
4. `training_telemetry.jsonl` integrity with verified mutation cycle logging.

### 2.2 Disambiguation of Method Variants
To prevent confusion in paper tables, the 23 runs are categorized into distinct architectural generations:

| Method Code | Pruning Lifecycle | Inflow Policy | Hard Budget | Population Cap | Scenes Tested | Physical Runs |
|---|---|---|:---:|:---:|---|:---:|
| **$V_{\text{ref}}$** | Vanilla 3DGS | Unconstrained | None | None | `bonsai` (S0) | 1 |
| **$A_0$** | Vanilla + Stats | Unconstrained | None | None | `bonsai` (S0) | 1 |
| **$A_1$ (V1)** | Naive Hard Budget | Unconstrained | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_1^{(G)}$ (V2)**| Decoupled + Grace | Unconstrained | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_1^{(V3)}$ (V3)**| Decoupled + Grace | Dynamic Inflow | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_2$** | Decoupled + Grace | Unconstrained | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_3$** | Structure Protected| Unconstrained | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_3^{(G)}$** | Protected + Grace | Unconstrained | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_3^{(V3)}$** | Protected + Grace | Dynamic Inflow | 6,144 MiB | 1.00M | `bonsai` (S0) | 1 |
| **$A_1^{(V4-1.15M)}$**| Decoupled Hygiene | Dynamic Inflow | 6,144 MiB | 1.15M | `bonsai` (S0, S1, S2), `garden`, `flowers` | 5 |
| **C1100, C1200, C1300**| Decoupled Hygiene | Dynamic Inflow | 6,144 MiB | 1.1M, 1.2M, 1.3M | `bonsai` (S0) | 3 |
| **`ABL-DI`** | Decoupled Hygiene | Static 5k Inflow | 6,144 MiB | 1.15M | `bonsai` (S0) | 1 |
| **`ABL-HP`** | **Preservation** | **Dynamic Inflow** | **6,144 MiB** | **1.15M** | **`bonsai` (S0, S1, S2), `garden`, `flowers`** | **5** |

### 2.3 Dataset Split Verification (LLFF Hold = 8)
Inspection of `scene/dataset_readers.py` confirms that test cameras are determined strictly by index:
$$\text{Test Views} = \{i \mid i \pmod 8 == 0\}$$

Auditing image directories directly against this rule:
- **`bonsai`:** 292 image files. Test indices: $0, 8, \dots, 288 \implies \mathbf{37\text{ holdout views}}$, $255\text{ train views}$.
- **`garden`:** 185 image files. Test indices: $0, 8, \dots, 184 \implies \mathbf{24\text{ holdout views}}$, $161\text{ train views}$.
- **`flowers`:** 173 image files. Test indices: $0, 8, \dots, 168 \implies \mathbf{22\text{ holdout views}}$, $151\text{ train views}$.

All raw evaluation files (`test/ours_30000/results.json`) confirm that exactly 37 views were evaluated on `bonsai`, 24 views on `garden`, and 22 views on `flowers`.

---

## 3. Forensic Audit of Candidate Accounting

### 3.1 Mathematical Derivation of Admission Partition
In `zerogs/admission_controller.py`, candidate requests are partitioned:
$$\text{admissible\_cap} = \min(N_{\text{computed\_cap}}, N_{\text{requested}})$$
$$\text{rejected\_by\_cap} = \max(0, N_{\text{requested}} - N_{\text{computed\_cap}})$$

Algebraic identity:
$$\text{admissible\_cap} + \text{rejected\_by\_cap} = N_{\text{requested}}$$
This mathematical partition invariant holds with zero residual ($\Delta = 0$) at the admission interface.

### 3.2 The Pre-Prune Reallocation Defect in Telemetry
During cycles where physical memory is tight ($H_{\text{usable}} \le 1,024\text{ MiB}$), the controller issues a `PRUNE_FIRST` action:
1. Low-utility primitives ($K$ items) are evicted via `gaussians.prune_points`.
2. PyTorch cache is evacuated via `torch.cuda.empty_cache()`.
3. In `train.py` (line 594), `post_prune_decision = admission_ctrl.evaluate_admission(...)` is called.
4. Because memory was freed, `post_prune_decision` computes a higher cap (up to $15,000$).
5. The Knapsack allocator executes densification up to this higher cap ($N_{\text{executed}} = 15,000$).
6. **Telemetry Defect:** Line 652 logged `decision.to_dict()` (the initial pre-prune decision with the 5,000 cap) instead of `allocation_decision.to_dict()` (the post-prune decision).

**Forensic Impact:**
This is an observational logging timing defect, **not a memory safety failure or primitive leak**. The physical GPU memory was strictly freed before the additional candidates were admitted.

---

## 4. Forensic Audit of GPU Memory Accounting

### 4.1 Deconstruction of the 1,137 Bytes/Gaussian Footprint
Audit of `zerogs/cost_model.py` and `experiments/results/cost_calibration.json` reveals:
- Base parameters (XYZ, rotation, scale, opacity, SH degree 3) = $59\text{ floats} \times 4\text{ B} = 236\text{ B}$.
- Adam optimizer states (Gradients + 2 moments) = $3 \times 236\text{ B} = 708\text{ B}$.
- **Base Persistent Memory per Gaussian:** $236 + 708 = \mathbf{944\text{ Bytes}}$.
- Empirical clone transient multiplier (from hardware profiling): $\mu_{\text{clone}} = 1.204584$.
- **Transient Clone Unit Cost:** $944\text{ B} \times 1.204584 = \mathbf{1{,}137.13\text{ Bytes}} \approx \mathbf{1{,}137\text{ B}}$.
- Empirical split transient multiplier: $\mu_{\text{split}} = 2.573835$.
- **Transient Split Unit Cost:** $944\text{ B} \times 2.573835 = \mathbf{2{,}429.70\text{ Bytes}} \approx \mathbf{2{,}429\text{ B}}$.

**Discrepancy Documented:** Early markdown files referred to $1{,}137\text{ B}$ as the persistent footprint. It is the transient cloning cost.

### 4.2 Hard Guarantee vs Empirical Boundedness
- The Admission Controller guarantees that the **densification step** will not exceed the effective budget.
- However, forward/backward rasterizer tile sorting spikes depend on viewpoint, focal zoom, and screen resolution.
- Under Windows 11 WDDM, OS display consumption fluctuates around ~1.8 GB.
- **Audit Verdict:** The framework provides **robust empirical memory boundedness with zero OOMs across 23 physical benchmark runs**, but claiming an unconditional mathematical "Hard Physical VRAM Guarantee" for arbitrary resolutions and viewpoints is **UNSUPPORTED**.

---

## 5. Audit of Scientific Claims & Evidence

Every core claim of the paper was evaluated and classified in [`CLAIM_EVIDENCE_MATRIX.csv`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/CLAIM_EVIDENCE_MATRIX.csv):

| Claim ID | Scientific Claim | Audit Status | Key Evidence & Confounders | Action for Paper |
|:---:|---|:---:|---|---|
| **CLM-01** | Dynamic Headroom-Aware Inflow Governor | **VERIFIED** | Throttled 8.91M requests to 985K in `flowers` (88.93% rejection), preventing OOM. | Retain as Core Contribution 1. |
| **CLM-02** | Marginal Knapsack Spatial Allocator | **VERIFIED** | Greedy knapsack ordered by $\lambda = \text{demand}/\text{cost}$ with $\kappa_{\text{split}} = 0.40$ verified. | Retain as Core Contribution 2. |
| **CLM-03** | Preservation Lifecycle / Over-Pruning Dilemma | **VERIFIED** | `ABL-HP` beats baseline V4 on all 3 scenes (+11.08 dB, +6.26 dB, +2.94 dB). Over-pruning mechanism confirmed. | Retain as Primary Empirical Breakthrough. |
| **CLM-04** | Candidate Conservation ($\Delta = 0$) | **PARTIALLY VERIFIED** | Holds at admission interface. Complicated by pre-prune reallocation telemetry logging defect. | Downgrade from global theorem to admission partition invariant. |
| **CLM-05** | Hard Physical VRAM Guarantee | **PARTIALLY VERIFIED** | 0 OOMs across 23 runs. Cannot guarantee rasterizer backward spikes for arbitrary views. | Downgrade to Empirical Boundedness with Zero OOMs. |
| **CLM-06** | Outperforming Unconstrained 3DGS ($V_{\text{ref}}$) | **PARTIALLY VERIFIED** | Verified on `bonsai` (+2.24 dB over $V_{\text{ref}}$). Untested on `garden` and `flowers`. | Restrict claim to `bonsai`. |
| **CLM-07** | Cross-Scene Generalizability | **VERIFIED** | Consistent gains on `bonsai`, `garden`, and `flowers` on 8GB GPU. | Retain cross-scene claim. Disclose single-seed execution for outdoor scenes. |
| **CLM-08** | Paired Multi-Seed Significance ($p = 0.0145$) | **VERIFIED** | Recomputed: $t(2) = 8.2110, p = 0.01451$, Cohen's $d = 4.7406$. | Retain statistics. Disclose $df=2$ sample size limitation. |

---

## 6. Numerical Discrepancy Matrix

All 10 numerical discrepancies uncovered across repository files are cataloged in [`NUMERICAL_DISCREPANCY_MATRIX.csv`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/NUMERICAL_DISCREPANCY_MATRIX.csv).

**Key Corrections Required in Manuscript:**
1. `paper/EXPERIMENTAL_RESULTS.md`: Correct `garden` holdout views from 23 to 24, and `flowers` from 21 to 22.
2. `paper/TABLES.md` Table 2: Replace the stale hypothetical placeholder row for ABL-HP (18.12 dB) with the true physical result (31.28 dB).
3. `paper/TABLES.md` Table 4: Replace synthetic placeholder multi-seed numbers for V4 (std 0.0189 dB) with the true empirical physical numbers ($20.7455 \pm 1.8690\text{ dB}$).
4. All documents: Clarify $944\text{ B}$ persistent footprint vs $1{,}137\text{ B}$ transient clone cost.

---

## 7. Answers to the 5 Mandatory Final Questions

### Question 1: Những kết quả nào đủ tin cậy để đưa vào paper?
**Trả lời:**
1. **Kết quả tái lập đa seed trên Bonsai ($n=3$):** Baseline V4 ($20.75 \pm 1.87\text{ dB}$) vs Đề xuất ABL-HP (**$31.83 \pm 0.48\text{ dB}$**), Paired gain **$+11.0850 \pm 2.3383\text{ dB}$**, $t(2) = 8.2110$, $p = 0.01451$, Cohen's $d = 4.7406$. Đầy đủ checkpoint, PLY, telemetry và kết quả đánh giá 37 holdout views.
2. **Kết quả kiểm chứng chéo trên Garden (Seed 0, 30k steps):** Baseline V4 ($18.6656\text{ dB}$, SSIM 0.3847) vs ABL-HP (**$24.9277\text{ dB}$**, SSIM 0.7455), Paired gain **$+6.2621\text{ dB}$**, Peak VRAM $7,570.0\text{ MiB}$ (0 OOM). Đầy đủ 24 holdout views.
3. **Kết quả kiểm chứng chéo trên Flowers (Seed 0, 30k steps):** Baseline V4 ($17.3572\text{ dB}$, SSIM 0.3378) vs ABL-HP (**$20.2994\text{ dB}$**, SSIM 0.5171), Paired gain **$+2.9422\text{ dB}$**, Peak VRAM $7,288.0\text{ MiB}$ (0 OOM), tiết chế $88.93\%$ candidates dưới áp lực 8.91M requests. Đầy đủ 22 holdout views.
4. **So sánh với Unconstrained $V_{\text{ref}}$ trên Bonsai:** ABL-HP đạt $31.83\text{ dB}$ vượt $V_{\text{ref}}$ ($29.59\text{ dB}$) **$+2.24\text{ dB}$** với **$40.2\%$ ít Gaussians hơn** ($738\text{K}$ vs $1.24\text{M}$) và peak VRAM thấp hơn ($7,338\text{ MiB}$ vs $7,656\text{ MiB}$).
5. **Độ an toàn bộ nhớ 8GB:** 23/23 physical runs hoàn thành trọn vẹn 30,000 steps mà không xảy ra bất kỳ lỗi OOM nào.

### Question 2: Những kết quả nào phải loại bỏ hoặc chạy lại?
**Trả lời:**
1. **Loại bỏ hoàn toàn Bảng 4 trong `paper/TABLES.md`:** Bảng này chứa số liệu đa seed giả định cũ ($22.8150$, $22.8520$ với std $0.0189\text{ dB}$). Phải thay thế bằng Bảng 2 trong `RECOMMENDED_FINAL_RESULTS.md` chứa số liệu vật lý thật ($19.2472$ và $20.1494$, std $1.8690\text{ dB}$).
2. **Loại bỏ dòng ABL-HP giả định cũ trong Bảng 2 của `paper/TABLES.md`:** Dòng cũ ghi $18.12\text{ dB}$ (từ thiết kế giả định trước khi chạy). Phải thay bằng $31.2776\text{ dB}$.
3. **Sửa các số liệu holdout view sai số học trong `paper/EXPERIMENTAL_RESULTS.md`:** Sửa `garden` từ 23 thành 24 views; sửa `flowers` từ 21 thành 22 views.

### Question 3: Còn thiếu thí nghiệm nào để chứng minh từng contribution?
**Trả lời:**
1. **Để chứng minh "Vượt trội Unconstrained 3DGS trên mọi scene":** Hiện còn thiếu run vật lý của $V_{\text{ref}}$ trên `garden` và `flowers`. Nếu không chạy 2 run này, bài báo phải giới hạn tuyên bố vượt $V_{\text{ref}}$ trên `bonsai`.
2. **Để bóc tách độc lập hoàn toàn 2x2 Factorial giữa Dynamic Inflow và Preservation Pruning:** Còn thiếu ô `ABL-DI-HP` (Static 5k Cap + Legacy Preservation Lifecycle) trên `bonsai`.
3. **Để tăng bậc tự do thống kê ($df > 2$):** Có thể mở rộng đa seed lên $n=5$ (chạy thêm Seed 3 và 4). Nếu không, phải ghi rõ giới hạn $df=2$ trong phần Limitations.

### Question 4: Những claim nào phải hạ mức độ?
**Trả lời:**
1. **Hạ mức độ "Hard Physical VRAM Guarantee":** Chuyển thành *"Headroom-Aware Bounded Densification with Zero Observed OOMs across 23 Physical Benchmarks"*.
2. **Hạ mức độ "Universal Global Candidate Conservation Theorem ($\Delta = 0$)":** Chuyển thành *"Deterministic Partition Invariant at the Admission Interface"*, kèm giải thích cơ chế tái phân bổ headroom sau pre-pruning.
3. **Hạ mức độ "Outperforming Vanilla 3DGS across all scenes":** Thu hẹp thành *"Outperforming Vanilla 3DGS on the complex bounded scene Bonsai while strictly bounded under 8GB VRAM"*.

### Question 5: Paper hiện tại đã đủ cơ sở để chuyển sang manuscript final chưa?
**Trả lời: ĐÃ ĐỦ CƠ SỞ ĐỂ CHUYỂN SANG MANUSCRIPT FINAL (với điều kiện áp dụng đúng các số liệu và điều chỉnh mức độ claim đã audit).**
Cốt lõi khoa học của bài báo là **hiện tượng Over-Pruning Dilemma** và phương pháp giải quyết bằng **Preservation-Centric Dynamic Inflow** đã được chứng minh thực nghiệm vật lý vững chắc trên cả 3 scene với mức tăng paired rất lớn ($+11.08\text{ dB}$, $+6.26\text{ dB}$, $+2.94\text{ dB}$), 100% không OOM, và kích thước hiệu ứng Cohen's $d = 4.74$. Khi thay thế các bảng biểu cũ bằng nội dung trong [`RECOMMENDED_FINAL_RESULTS.md`](file:///C:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/experiments/results/RECOMMENDED_FINAL_RESULTS.md) và tích hợp các lưu ý kiểm toán, bản thảo hoàn toàn đạt tiêu chuẩn xuất bản tại các hội nghị thị giác máy tính hàng đầu.
