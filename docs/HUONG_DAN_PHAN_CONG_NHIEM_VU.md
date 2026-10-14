# KẾ HOẠCH VÀ HƯỚNG DẪN TRIỂN KHAI ĐỀ TÀI ZEROGS
## Phân công Nhiệm vụ Nghiên cứu & Phát triển cho 4 Thành viên

**Đề tài:** ZeroGS: Byte-Level Transient Memory Modeling and Occlusion-Aware Admission Control for 3D Gaussian Splatting  
**Mục tiêu NCKH:** Xây dựng framework huấn luyện 3DGS bị giới hạn bộ nhớ VRAM, đảm bảo **Zero-OOM (0% crash)** và khắc phục triệt để **tụt giảm 0.20 – 0.28 dB PSNR** trên các cảnh bị che khuất phức tạp.  
**Nhóm thực hiện:** Tường Huy, Quang Huy, Vinh, Phong.

---

## 🏗️ 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG & SƠ ĐỒ PHỐI HỢP

Đề tài giải quyết 2 bài toán khoa học:
1. **Bài toán Đột biến bộ nhớ tức thời (Transient Memory Spikes):** Toán tử Split sinh ra đỉnh bộ nhớ gấp **2.1×** so với Clone trên GPU thực tế, gây tràn RAM GPU ảo (OOM) nếu chỉ kiểm soát theo số lượng hạt thô sơ ($N$).
2. **Bài toán Xóa nhầm cấu trúc mảnh do che khuất (Occlusion Over-Pruning):** Đánh giá độ quan trọng theo số lượt nhìn thấy ($V_i$) ngây thơ khiến tán cây, cành lá mảnh (*bonsai*, *flowers*) bị tỉa nhầm, làm sụt giảm chất lượng hình ảnh.

```
                         [ Camera Frustums / Viewpoints ]
                                        │
                                        ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ 3. VINH: Occlusion-Aware Demand Engine (Gap S1)                          │
   │    - Trích xuất độ truyền qua ánh sáng (T_bar) và phương sai sâu (σ_z)   │
   │    - Tính điểm nhu cầu bù trừ che khuất: D_tilde = H * V * Ω             │
   │    - Sinh mặt nạ bảo vệ hình học mảnh (Structural Protection Mask)       │
   └───────────────────────────────┬──────────────────────────────────────────┘
                                   │
               D_tilde             │           Protection Mask
                                   ▼                  │
   ┌──────────────────────────────────────────┐       │
   │ 4. PHONG: Marginal Utility Allocator     │       │
   │    - Knapsack ratio: λ = D_tilde / Byte  │       │
   │    - Ưu tiên hạt có độ giảm loss cao nhất│       │
   └───────────────────┬──────────────────────┘       │
                       │                              │
          (N_clone, N_split)                          │
                       ▼                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ 1. TƯỜNG HUY: Byte-Level Cost Model (Gaps P1, S3)                        │
   │    - Dự báo chính xác byte tĩnh (944 B/pt) & đỉnh đột biến tức thời       │
   │    - ΔM_transient = N_c * B * φ_c + N_s * B * φ_s                        │
   └───────────────────────────────┬──────────────────────────────────────────┘
                                   │
                           ΔM_transient
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ 2. QUANG HUY: Admission Controller & Prune-First (Gaps P2, D2)           │
   │    - Đọc Headroom thực tế từ CUDA Allocator: Headroom = B_VRAM - M_alloc  │
   │    - Quyết định: ACCEPT / PRUNE_FIRST / SHRINK_AND_PRUNE_FIRST           │
   │    - Tỉa trước (Prune-First) K_prune hạt -> Dọn cache -> Thực thi Clone/Split│
   └───────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
          [ Vòng lặp Huấn luyện An toàn - 0% OOM - Tối ưu PSNR ]
```

---

## 📋 2. BẢN HƯỚNG DẪN CHI TIẾT TỪNG THÀNH VIÊN

---

### 👤 THÀNH VIÊN 1: TƯỜNG HUY
**Chuyên trách:** Mô hình hóa bộ nhớ cấp độ Byte & Thực nghiệm Đột biến tức thời (Hardware & Transient Memory Modeling)  
**Research Gaps giải quyết:** Gap P1 (Transient Spikes) & Gap S3 (Cost Modeling Disparity).

#### 1. Làm cái gì? (What)
* **Lý thuyết nền tảng:**
  - Tính toán chính xác dung lượng của 1 hạt Gaussian bậc SH $d=3$:
    $$\text{Vị trí } (3) + \text{Phép quay } (4) + \text{Tỷ lệ } (3) + \text{Độ mờ } (1) + \text{Hệ số màu SH } (48) = 59 \text{ số thực float32}$$
    $$B_{param} = 59 \times 4 \text{ Bytes} = 236 \text{ Bytes/Gaussian}$$
  - Dung lượng tĩnh khi huấn luyện (Tham số + Gradient + 2 Moment Adam `exp_avg`, `exp_avg_sq`):
    $$B_{persistent} = B_{param} + B_{grad} + 2 \times B_{param} = 944 \text{ Bytes/Gaussian}$$
  - Đột biến động (Transient Spike): Khi chạy `densify_and_split`, GPU phải tạm giữ $N_{split}$ hạt cha mẹ, sinh đồng thời $2N_{split}$ hạt con, và cấp phát lại toàn bộ bảng trạng thái Adam. Vì vậy, Split tạo đỉnh bộ nhớ cao gấp $\approx 2.1\times$ so với Clone.
* **Mã nguồn liên quan:**
  - `zerogs/cost_model.py` (Class `ByteCostModel`)
  - `zerogs/mock_gaussian_model.py` (Mô phỏng footprint Adam của 3DGS)
  - `experiments/profile_clone_vs_split.py` (Kịch bản đo đạc thực nghiệm GPU)
  - `tests/test_cost_model.py` (Unit tests)

#### 2. Làm như thế nào? (How - Từng bước kỹ thuật)
1. **Bước 1 — Viết module `ByteCostModel` (`zerogs/cost_model.py`):**
   - Cài đặt phương thức `calculate_persistent_bytes(n_points: int) -> int` trả về $n \times 944$.
   - Cài đặt phương thức `predict_mutation_transient_bytes(n_clone: int, n_split: int) -> int`:
     $$\Delta M_{transient} = n_{clone} \cdot B_{persistent} \cdot \phi_{clone} + n_{split} \cdot B_{persistent} \cdot \phi_{split}$$
   - Hiệu chỉnh các hệ số trên phần cứng NVIDIA: $\phi_{clone} = 1.15$, $\phi_{split} = 2.40$.
2. **Bước 2 — Thực nghiệm đo đạc phần cứng (Experiment 0):**
   - Viết kịch bản `experiments/profile_clone_vs_split.py`.
   - Sử dụng CUDA Memory Profiling API của PyTorch:
     ```python
     torch.cuda.reset_peak_memory_stats(device)
     # Thực thi Clone / Split
     peak_bytes = torch.cuda.max_memory_allocated(device)
     ```
   - Chạy với các batch kích thước $\Delta N \in \{10k, 50k, 100k, 200k\}$ trên GPU thật.
   - Vẽ biểu đồ đối sánh cột (Bar chart) lưu vào `experiments/results/experiment_0_transient_spikes.png`.
3. **Bước 3 — Viết Unit Test:**
   - Viết test trong `tests/test_cost_model.py` để kiểm tra độ lệch dự báo của `ByteCostModel` so với thực tế $< 5\%$.

#### 3. Mục tiêu hoàn thành (Deliverables)
- [ ] Hoàn thiện class `ByteCostModel` chuẩn type hints và docstrings.
- [ ] Báo cáo thực nghiệm `docs/EXPERIMENT_0_REPORT.md` kèm biểu đồ chứng minh hệ số $2.1\times$.
- [ ] 100% test pass lệnh: `pytest tests/test_cost_model.py`.

---

### 👤 THÀNH VIÊN 2: QUANG HUY
**Chuyên trách:** Bộ điều khiển tiếp nhận & Cơ chế Tỉa-Trước đảm bảo Zero-OOM (Admission Controller & Prune-First Protocol)  
**Research Gaps giải quyết:** Gap P2 (OOM Under Hard Limits) & Gap D2 (Order-of-Execution Vulnerability).

#### 1. Làm cái gì? (What)
* **Lý thuyết nền tảng:**
  - Các hệ thống cũ gặp lỗi OOM vì chúng tạo hạt mới *trước*, rồi mới dọn dẹp hạt cũ *sau* (Clone/Split $\to$ Prune). Tại thời điểm giao thời, bộ nhớ bị cộng dồn cả cũ lẫn mới, vượt trần VRAM.
  - Giải pháp của ZeroGS là **Prune-First (Dọn dẹp trước)**:
    - Đọc khoảng trống bộ nhớ an toàn: $\text{Headroom} = \max(0, B_{VRAM} - \delta_{safety} - M_{allocated})$.
    - **3 kịch bản quyết định tiếp nhận:**
      1. `ACCEPT`: Nếu $\Delta M_{transient} \le \text{Headroom}$.
      2. `PRUNE_FIRST`: Nếu thiếu dung lượng nhưng số hạt dư thừa khả dụng $N_{redundant}$ đủ để giải phóng:
         $$K_{prune} = \left\lceil \frac{\Delta M_{transient} - \text{Headroom}}{B_{persistent}} \right\rceil$$
         Tỉa ngay $K_{prune}$ hạt và gọi `torch.cuda.empty_cache()` trước khi cấp phát mới.
      3. `SHRINK_AND_PRUNE_FIRST`: Nếu dọn hết hạt thừa vẫn không đủ, co cụm số lượng hạt dự định sinh theo hệ số $\alpha = \frac{\text{Headroom} + \Delta M_{freed}}{\Delta M_{transient}}$.
* **Mã nguồn liên quan:**
  - `zerogs/admission_controller.py` (Class `AdmissionController`)
  - `experiments/stress_test_admission.py` (Kịch bản Stress Test OOM)
  - `tests/test_admission_controller.py` (Unit tests)

#### 2. Làm như thế nào? (How - Từng bước kỹ thuật)
1. **Bước 1 — Viết module `AdmissionController` (`zerogs/admission_controller.py`):**
   - Khởi tạo nhận `cost_model: ByteCostModel`, `hard_vram_limit_mb: float`, `safety_headroom_mb: float`.
   - Cài đặt hàm `evaluate_admission(n_clone, n_split, n_redundant) -> AdmissionDecision`:
     - Trả về đối tượng dataclass chứa: `action` (`ACCEPT`, `PRUNE_FIRST`, `SHRINK`), `k_prune`, `admitted_clone`, `admitted_split`.
   - Cài đặt hàm `execute_safe_mutation(...)`: Đảm bảo quy trình tuần tự:
     `Prune K_prune hạt` $\to$ `torch.cuda.empty_cache()` $\to$ `Clone & Split`.
2. **Bước 2 — Thực nghiệm kiểm thử sức ép OOM (Experiment 1):**
   - Viết file `experiments/stress_test_admission.py`.
   - Thiết lập kịch bản huấn luyện liên tục đẩy số hạt lên cao dưới các mức trần VRAM ngặt nghèo: 300MB, 500MB, 800MB, 4GB, 6GB.
   - So sánh tỷ lệ crash giữa Vanilla 3DGS (crash 100%) và ZeroGS (duy trì 0% crash).
   - Xuất biểu đồ so sánh: `experiments/results/stress_test_comparison.png`.
3. **Bước 3 — Viết Unit Test:**
   - Viết các test case trong `tests/test_admission_controller.py` kiểm tra các nhánh `ACCEPT`, `PRUNE_FIRST`, và nhánh co cụm `SHRINK`.

#### 3. Mục tiêu hoàn thành (Deliverables)
- [ ] Hoàn thiện class `AdmissionController` với logic kiểm soát bộ nhớ chính xác.
- [ ] Báo cáo thực nghiệm chứng minh tỷ lệ 0% OOM trên mọi mức trần VRAM.
- [ ] 100% test pass lệnh: `pytest tests/test_admission_controller.py`.

---

### 👤 THÀNH VIÊN 3: VINH
**Chuyên trách:** Động cơ Bù trừ Che khuất & Bảo vệ Cấu trúc Mảnh (Occlusion Engine & Structural Protection)  
**Research Gaps giải quyết:** Gap S1 (Occlusion-Induced Quality Drops & Foliage Over-Pruning).

#### 1. Làm cái gì? (What)
* **Lý thuyết nền tảng:**
  - Trong phương pháp SPARE-GS, độ quan trọng tính theo $D_i = H_i \cdot V_i$ (Gradient $\times$ Số lần nhìn thấy). Với các cảnh có nhiều tán cây, bông hoa (*bonsai*, *flowers*, *garden*), góc nhìn camera thay đổi khiến các chi tiết mảnh bị che khuất tạm thời ở một số view, làm $V_i$ bị giảm giả tạo $\to$ bị xếp vào đáy danh sách và bị xóa nhầm, gây sụt giảm $0.20 – 0.28\text{ dB PSNR}$.
  - Giải pháp ZeroGS trích xuất 2 giá trị vật lý từ rasterizer:
    1. Độ truyền qua ánh sáng tích lũy $\bar{T}_i \in [0, 1]$ (hạt càng bị che chắn nhiều thì $T$ càng nhỏ).
    2. Phương sai chiều sâu $\sigma_{z, i}$ trên các góc camera khác nhau.
  - **Công thức bù trừ nhu cầu:**
    $$\Omega_i = 1.0 + \gamma_{occ} \cdot \frac{\sigma_{z, i}}{\text{scene\_extent} + \epsilon} \cdot (1.0 - \bar{T}_i)$$
    $$\widetilde{D}_i = H_i \cdot V_i \cdot \Omega_i$$
  - **Mặt nạ bảo vệ cấu trúc (Structural Protection Mask):**
    Các hạt có $\sigma_{z, i} > 0.01 \cdot \text{scene\_extent}$ và $\bar{T}_i < \tau_{trans}$ được đánh dấu bảo vệ đặc biệt, **cấm tuyệt đối không được tỉa**.
* **Mã nguồn liên quan:**
  - `zerogs/occlusion_engine.py` (Class `OcclusionAwareEngine`)
  - `submodules/diff-gaussian-rasterization/` (Tùy biến hoặc hook lấy transmittance & depth_var)
  - `experiments/ablation_occlusion.py` (Thực nghiệm Ablation trên scene bonsai/flowers)
  - `tests/test_occlusion_engine.py` và `tests/test_cuda_rasterizer_stats.py`

#### 2. Làm như thế nào? (How - Từng bước kỹ thuật)
1. **Bước 1 — Viết module `OcclusionAwareEngine` (`zerogs/occlusion_engine.py`):**
   - Cài đặt phương thức `update_view_stats(transmittance, depth_var, visibility_filter)` để cộng dồn số liệu qua các camera viewpoint.
   - Cài đặt phương thức `compute_compensated_demand(radii, grads, scene_extent)` xuất ra tensor $\widetilde{D}$.
   - Cài đặt phương thức `generate_protection_mask()` trả về mask boolean đánh dấu các hạt không được tỉa.
2. **Bước 2 — Tích hợp lấy số liệu từ Rasterizer:**
   - Đảm bảo trong hàm render của hệ thống, khi forward pass diễn ra, các tensor phụ trợ `transmittance` và `depth_var` được ghi nhận đúng shape $[N]$ và chuyển an toàn sang `OcclusionAwareEngine`.
3. **Bước 3 — Thực nghiệm kiểm chứng Ablation (Experiment 2):**
   - Viết file `experiments/ablation_occlusion.py`.
   - Chạy trên scene nhạy cảm che khuất (*bonsai*):
     - Bản chạy 1: Không có Occlusion Engine (Baseline).
     - Bản chạy 2: Có $\Omega_i$ và Structural Protection Mask.
   - So sánh số lượng hạt tán lá giữ lại được và chứng minh chất lượng phục hồi tăng $+0.28\text{ dB PSNR}$.
4. **Bước 4 — Viết Unit Test:**
   - Viết test trong `tests/test_occlusion_engine.py` khẳng định các hạt thuộc mask bảo vệ không bao giờ rơi vào danh sách `prune_candidates`.

#### 3. Mục tiêu hoàn thành (Deliverables)
- [ ] Hoàn thiện class `OcclusionAwareEngine` với đầy đủ thuật toán bù trừ che khuất.
- [ ] Kết quả thực nghiệm Ablation chứng minh cứu được chi tiết tán lá và cành cây mảnh.
- [ ] 100% test pass lệnh: `pytest tests/test_occlusion_engine.py`.

---

### 👤 THÀNH VIÊN 4: PHONG
**Chuyên trách:** Phân bổ Quota Cận biên & Tích hợp Hệ thống, Benchmark Toàn diện (Marginal Allocator & System Integration / Benchmark)  
**Research Gaps giải quyết:** Gap P3 (Suboptimal Count-Based Quota) & Tích hợp End-to-End.

#### 1. Làm cái gì? (What)
* **Lý thuyết nền tảng:**
  - Phân bổ hạn mức hạt bằng số lượng thô ($N$) là không tối ưu vì mỗi thao tác Split tiêu tốn nhiều bộ nhớ hơn Clone nhưng có thể mang lại mức giảm loss khác nhau.
  - ZeroGS chuyển bài toán thành dạng **Túi ba lô (Knapsack problem)** trên đơn vị Byte:
    $$\lambda_i^{clone} = \frac{\widetilde{D}_i}{B_{persistent} \cdot \phi_{clone}}, \quad \lambda_j^{split} = \frac{\widetilde{D}_j}{B_{persistent} \cdot \phi_{split}}$$
  - Sắp xếp thứ tự ưu tiên giảm dần theo $\lambda$: Tối đa hóa lượng cải thiện chất lượng trên từng Megabyte VRAM đầu tư.
  - **Tích hợp toàn diện:** Kết nối cả 3 module của Tường Huy, Quang Huy, Vinh vào vòng lặp huấn luyện chính thông qua `MemoryBoundedDensificationScheduler`, điều phối các script `train.py`, `render.py`, `metrics.py`, và chạy bộ benchmark tổng thể.
* **Mã nguồn liên quan:**
  - `zerogs/marginal_allocator.py` (Class `MarginalUtilityAllocator`)
  - `zerogs/scheduler.py` (`MemoryBoundedDensificationScheduler`)
  - `train.py`, `render.py`, `metrics.py`
  - `experiments/benchmark_suite.py` (Multi-Scene Benchmark Suite)
  - `tests/test_marginal_allocator.py` và `tests/test_scheduler.py`

#### 2. Làm như thế nào? (How - Từng bước kỹ thuật)
1. **Bước 1 — Viết module `MarginalUtilityAllocator` (`zerogs/marginal_allocator.py`):**
   - Nhận tensor $\widetilde{D}$ từ module của **Vinh** và hệ số $\phi_{clone}, \phi_{split}$ từ module của **Tường Huy**.
   - Cài đặt hàm `allocate_quota(candidates_clone, candidates_split, allowed_bytes)`:
     Sắp xếp theo $\lambda$, greedy chọn cho đến khi chạm `allowed_bytes`.
2. **Bước 2 — Hiện thực `MemoryBoundedDensificationScheduler` (`zerogs/scheduler.py`):**
   - Kết nối 4 khối: Mỗi chu kỳ densification (ví dụ iteration 500 đến 15000, chu kỳ 100 step):
     1. Gọi Vinh: lấy điểm $\widetilde{D}$ và `protection_mask`.
     2. Gọi Phong: tính toán ứng viên Clone và Split theo Knapsack.
     3. Gọi Tường Huy: tính $\Delta M_{transient}$.
     4. Gọi Quang Huy: duyệt quota và thực hiện quy trình `Prune-First` an toàn.
   - Tích hợp cờ cấu hình vào `train.py`: `--hard_vram_limit_mb`, `--safety_headroom_mb`, `--gamma_occ`.
3. **Bước 3 — Thực hiện Multi-Scene Benchmark Suite (Phase 4):**
   - Viết và chạy `experiments/benchmark_suite.py` trên các scene chuẩn (Mip-NeRF 360: *garden*, *bonsai*, *bicycle*; Tanks & Temples).
   - Đo lường đầy đủ 5 tiêu chí:
     1. Peak VRAM (GB)
     2. Tỷ lệ OOM (%)
     3. PSNR (dB)
     4. SSIM
     5. LPIPS & Render FPS
   - Xuất bảng số liệu LaTeX `experiments/results/benchmark_table.tex`.
   - Vẽ đường cong Pareto Frontier: `experiments/results/pareto_psnr_vs_vram.png` (PSNR vs Peak VRAM) khẳng định ZeroGS vượt trội hoàn toàn so với 3DGS, Diet-GS, SPARE-GS.
4. **Bước 4 — Kiểm thử tích hợp toàn bộ:**
   - Chạy lệnh toàn diện: `python -m pytest tests/` đảm bảo toàn bộ test case đều xanh.

#### 3. Mục tiêu hoàn thành (Deliverables)
- [ ] Hoàn thiện class `MarginalUtilityAllocator` và bộ điều phối `scheduler.py`.
- [ ] Script huấn luyện `train.py` chạy ổn định, tự động dừng/thích ứng dưới giới hạn VRAM.
- [ ] Bảng số liệu LaTeX và đồ thị Pareto Frontier phục vụ trực tiếp cho bài báo khoa học.
- [ ] 100% test pass trên toàn bộ test suite dự án.

---

## 🤝 3. BẢNG GIAO ƯỚC DỮ LIỆU GIỮA CÁC THÀNH VIÊN (API CONTRACTS)

Để tránh xung đột khi code độc lập, 4 thành viên thống nhất các hàm giao tiếp sau:

| Bên cung cấp | Bên nhận | Tên Hàm / Phương Thức | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Tường Huy** | Quang Huy | `ByteCostModel.predict_mutation_transient_bytes` | `(n_clone: int, n_split: int)` | `int` (Số byte đột biến dự báo) |
| **Tường Huy** | Quang Huy & Phong | `ByteCostModel.calculate_persistent_bytes` | `(n_points: int)` | `int` (Dung lượng tĩnh $N \times 944$) |
| **Vinh** | Phong | `OcclusionAwareEngine.compute_compensated_demand` | `(radii: Tensor, grads: Tensor, extent: float)` | `Tensor` ($\widetilde{D}$, shape $[N]$) |
| **Vinh** | Quang Huy | `OcclusionAwareEngine.generate_protection_mask` | `()` | `BoolTensor` (shape $[N]$, True = Cấm tỉa) |
| **Quang Huy** | Phong | `AdmissionController.evaluate_admission` | `(n_clone: int, n_split: int, n_redundant: int)` | `AdmissionDecision` dataclass |
| **Phong** | Cả nhóm | `MemoryBoundedDensificationScheduler.step` | `(iteration: int, model: GaussianModel)` | None (Điều phối toàn bộ vòng lặp an toàn) |

---

## ⏱️ 4. KẾ HOẠCH TRIỂN KHAI 4 TUẦN (SPRINT ROADMAP)

| Mốc thời gian | Tường Huy | Quang Huy | Vinh | Phong |
| :--- | :--- | :--- | :--- | :--- |
| **Tuần 1: Khởi động & Nền tảng** | Đo lường thực tế trên GPU, viết `ByteCostModel` | Thiết kế khung `AdmissionController` và kịch bản stress test | Tìm hiểu kênh `transmittance` & `depth_var` từ renderer | Thiết lập môi trường, chuẩn bị pipeline metrics và test runner |
| **Tuần 2: Lõi thuật toán** | Hoàn thành `Experiment 0`, chốt các hệ số $\phi_{clone}, \phi_{split}$ | Hoàn thành cơ chế `Prune-First` và quyết định 3 trạng thái | Hoàn thiện công thức $\Omega_i$ và mặt nạ bảo vệ cấu trúc | Xây dựng thuật toán Knapsack `MarginalUtilityAllocator` |
| **Tuần 3: Tích hợp & Stress Test** | Hỗ trợ kiểm thử độ lệch bộ nhớ khi chạy thực tế | Chạy `stress_test_admission.py` khẳng định 0% OOM | Chạy `ablation_occlusion.py` khẳng định tăng +0.28 dB PSNR | Ghép 4 module vào `scheduler.py` và hook vào `train.py` |
| **Tuần 4: Đánh giá & Báo cáo** | Viết phần Phương pháp mô hình hóa bộ nhớ trong bài báo | Viết phần Hệ thống & Cơ chế kiểm soát tiếp nhận | Viết phần Phân tích vật lý che khuất & Ablation | Chạy toàn bộ Benchmark, vẽ Pareto, xuất bảng LaTeX |

---

## 🚀 5. HƯỚNG DẪN DÀNH CHO CẢ NHÓM BẮT ĐẦU NGAY

1. **Clone repo và cài đặt môi trường:**
   ```bash
   git clone https://github.com/ptanh05/ZeroGS.git
   cd ZeroGS
   pip install -r requirements.txt
   pip install -e .
   ```
2. **Chạy kiểm thử toàn bộ hệ thống:**
   ```bash
   python -m pytest -o pythonpath="." tests/
   ```
3. **Các lệnh chạy thử nghiệm nhanh:**
   ```bash
   # Tường Huy:
   python experiments/profile_clone_vs_split.py

   # Quang Huy:
   python experiments/stress_test_admission.py

   # Vinh:
   python experiments/ablation_occlusion.py

   # Phong:
   python experiments/benchmark_suite.py
   ```
