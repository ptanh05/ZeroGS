# VRAM BUDGET FALSE-POSITIVE FIX REPORT — PV-SRA-3DGS

**Ngày thực hiện:** 2026-11-03 22:36:00 (Local Time)  
**Tác vụ:** Điều tra và khắc phục lỗi False-Positive `VramBudgetExceededError` trong `assert_current_memory_within_budget()`.  
**Phần cứng kiểm chứng:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MB VRAM).

---

## 1. ROOT CAUSE ĐÃ XÁC MINH (VERIFIED ROOT CAUSE)

### Bối cảnh lỗi tại GPU Smoke Test Iteration 146:
Khi huấn luyện scene `data/bonsai` (`images_4`), tiến trình bị dừng tại iteration 146 với thông báo:
```text
VramBudgetExceededError: Current CUDA footprint exceeds the hard VRAM budget before mutation: footprint=6166.00MB, hard_budget=6144.00MB
```
Bản ghi telemetry tại thời điểm crash:
- `allocated_mb`: **2,861.79 MB**
- `peak_allocated_mb`: **3,338.05 MB**
- `reserved_mb`: **5,910.00 MB**
- `conservative_footprint_mb`: **5,910.00 MB**
- `non_pytorch_allowance_mb`: **256.00 MB**
- `hard_budget_mb`: **6,144.00 MB**

### Nguyên nhân kỹ thuật cốt lõi:
1. **Cơ chế Caching Allocator của PyTorch**:
   PyTorch quản lý VRAM theo cơ chế phân bổ khối (memory pool). Khi các bộ đệm rasterizer và tensor trung gian của backward pass được giải phóng (`del` hoặc out-of-scope), PyTorch **không gọi `cudaFree`** để trả bộ nhớ về hệ điều hành, mà giữ lại các khối này trong `torch.cuda.memory_reserved()`. Vùng nhớ chênh lệch giữa reserved và allocated ($5,910 - 2,862 = 3,048$ MB) thực chất là **bộ nhớ đệm tự do (free cached blocks)** hoàn toàn sẵn sàng cho PyTorch tái sử dụng mà không cần cấp phát thêm từ driver GPU.
2. **Khuyết tật logic trong `assert_current_memory_within_budget()` cũ**:
   Hàm tính `footprint = max(allocated, reserved) + non_pytorch_allowance`. Khi `reserved` tăng dần qua các camera view có kích thước khác nhau và chạm 5,910 MB, hàm đã cộng 256 MB allowance thành 6,166 MB (> 6,144 MB) và **lập tức ném ngoại lệ dừng huấn luyện** mà không kiểm tra xem phần vượt ngưỡng có phải là cache có thể thu hồi hay không.
3. **Phân tích khả năng thu hồi của `empty_cache()`**:
   - **Thu hồi được**: Các khối bộ nhớ không chứa tensor hoạt động (`inactive blocks`) trong pool của allocator. Thực nghiệm trên RTX 5060 chứng minh `empty_cache()` lập tức đưa `reserved` từ 5,910 MB về 2,861 MB.
   - **Không thu hồi được**: Bộ nhớ đang giữ tensor (`allocated_bytes`), phân mảnh nội tại trong khối đang chia tách (`inactive_split_bytes`), và bộ nhớ cấp phát ngoài PyTorch (đã được bảo vệ bằng `non_pytorch_allowance = 256 MB`).

---

## 2. CÁC FILE ĐÃ THAY ĐỔI

1. [`zerogs/admission_controller.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/zerogs/admission_controller.py):
   - Nâng cấp `VramBudgetExceededError` để lưu trữ snapshot `before_cleanup` và `after_cleanup`.
   - Bổ sung trường `self.total_cache_reclaims = 0`.
   - Tái cấu trúc hàm `assert_current_memory_within_budget()` với cơ chế Safe Cache Reclamation (thu hồi cache đúng một lần khi vượt trần, ghi telemetry và chỉ raise khi bộ nhớ sau thu hồi vẫn vượt budget).
2. [`train.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/train.py):
   - Truyền `telemetry_writer=telemetry_writer, iteration=iteration` vào `assert_current_memory_within_budget()`.
3. [`tests/test_vram_cache_reclamation.py`](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKH-2026-2027/ZeroGS/tests/test_vram_cache_reclamation.py):
   - Bộ test mới bao phủ toàn bộ 5 kịch bản A, B, C, D, E.

---

## 3. GIẢI THÍCH LOGIC TRƯỚC VÀ SAU KHI SỬA

### Trước khi sửa:
```python
# CŨ: Đánh giá mù quáng, coi toàn bộ reserved memory là VRAM không thể thu hồi
snapshot = capture_cuda_memory()
footprint = snapshot.conservative_footprint_bytes + self.non_pytorch_allowance_bytes
if footprint > self.hard_vram_limit_bytes:
    raise VramBudgetExceededError(...)
```

### Sau khi sửa:
```python
# MỚI: Quy trình Safe Cache Reclamation 2 giai đoạn
snapshot = capture_cuda_memory()
footprint = snapshot.conservative_footprint_bytes + self.non_pytorch_allowance_bytes

# Giai đoạn 1: Nếu dưới budget -> Tiếp tục bình thường, không tốn chi phí empty_cache
if footprint <= self.hard_vram_limit_bytes:
    return snapshot.to_dict()

# Giai đoạn 2: Vượt budget -> Thử thu hồi cache đúng 1 lần
if telemetry_writer is not None:
    telemetry_writer.write("cache_reclaim_triggered", ...)

reclaimed_bytes = 0
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Thu hồi các block inactive
    post_snapshot = capture_cuda_memory()
    post_footprint = post_snapshot.conservative_footprint_bytes + self.non_pytorch_allowance_bytes
    reclaimed_bytes = max(0, snapshot.reserved_bytes - post_snapshot.reserved_bytes)
else:
    post_snapshot, post_footprint = snapshot, footprint

# Nếu sau khi thu hồi mà footprint thực sự vẫn vượt trần -> Raise ngoại lệ thật
if post_footprint > self.hard_vram_limit_bytes:
    if telemetry_writer is not None:
        telemetry_writer.write("budget_exceeded_post_cleanup", ...)
    raise VramBudgetExceededError(..., before_cleanup=snapshot.to_dict(), after_cleanup=post_snapshot.to_dict())

# Thu hồi thành công -> Lưu sự kiện telemetry và tiếp tục an toàn
self.total_cache_reclaims += 1
if telemetry_writer is not None:
    telemetry_writer.write("cache_reclaimed", iteration=iteration, reclaimed_mb=..., ...)
return result
```

---

## 4. CÁC CHỈ SỐ BỘ NHỚ ĐƯỢC SỬ DỤNG

- **`allocated_bytes`** (`torch.cuda.memory_allocated`): Bộ nhớ thực sự gắn với các tensor đang tồn tại (weights, activations, optimizer states). **Không thể thu hồi** bởi `empty_cache()`.
- **`reserved_bytes`** (`torch.cuda.memory_reserved`): Tổng dung lượng bộ nhớ mà allocator đã lấy từ GPU driver.
- **`reclaimable_cache_bytes`** ($\approx \text{reserved} - \text{allocated}$): Bộ nhớ đệm không sử dụng, **có thể thu hồi** được bằng `torch.cuda.empty_cache()`.
- **`non_pytorch_allowance_bytes`**: Khoản đệm dự phòng cố định (256 MB) cho CUDA runtime context, WDDM driver hiển thị và các tiến trình nền.
- **`conservative_footprint_bytes`**: $\max(\text{allocated}, \text{reserved})$.

---

## 5. KẾT QUẢ KIỂM THỬ TỪNG TEST CASE (A – F)

Chạy lệnh kiểm thử:
```bash
python -m pytest -o pythonpath=. tests/test_vram_cache_reclamation.py -s -v
```

| Test Case | Kịch bản kiểm thử | Hành vi kỳ vọng | Kết quả thực tế |
| :---: | :--- | :--- | :---: |
| **Test A** | Normal footprint ($3,256 \le 6,144$ MB) | Không gọi `empty_cache`, không raise error, không ghi reclaim event. | **PASSED** |
| **Test B** | Recoverable cache ($6,156 > 6,144$ MB; sau cleanup $3,056 \le 6,144$ MB) | Gọi `empty_cache` đúng 1 lần, tiếp tục bình thường, ghi nhận `cache_reclaimed` (3,100 MB). | **PASSED** |
| **Test C** | Unrecoverable memory ($6,456 > 6,144$ MB cả trước và sau cleanup) | Gọi `empty_cache` 1 lần, raise `VramBudgetExceededError`, ghi đầy đủ before/after telemetry. | **PASSED** |
| **Test D** | Active allocations vượt budget ($6,200 > 6,144$ MB allocated) | Không bị đánh lừa bởi `empty_cache`, bắt buộc phải raise error. | **PASSED** |
| **Test E** | CUDA unavailable / CPU mock (`cuda.is_available() == False`) | Không gọi `empty_cache`, xử lý graceful trên CPU mock, raise đúng khi mock vượt trần. | **PASSED** |
| **Test F** | Regression toàn bộ test suite repository | Chạy toàn bộ 73 unit & integration tests trong thư mục `tests/`. | **73 / 73 PASSED** (5.36s) |

---

## 6. TÓM TẮT GIT DIFF

```diff
--- a/zerogs/admission_controller.py
+++ b/zerogs/admission_controller.py
@@ -20,2 +20,8 @@ class VramBudgetExceededError(RuntimeError):
     """Raised when the existing workload already exceeds the effective budget."""
+    def __init__(self, message: str, before_cleanup=None, after_cleanup=None):
+        super().__init__(message)
+        self.before_cleanup = before_cleanup
+        self.after_cleanup = after_cleanup
@@ -115,10 +121,55 @@ class AdmissionController:
-    def assert_current_memory_within_budget(self) -> Dict[str, Any]:
+    def assert_current_memory_within_budget(self, telemetry_writer=None, iteration=None):
         snapshot = capture_cuda_memory()
         footprint = snapshot.conservative_footprint_bytes + self.non_pytorch_allowance_bytes
+        if footprint <= self.hard_vram_limit_bytes:
+            return snapshot.to_dict()
+        ...
+        if torch.cuda.is_available():
+            torch.cuda.empty_cache()
+            post_snapshot = capture_cuda_memory()
+            post_footprint = post_snapshot.conservative_footprint_bytes + self.non_pytorch_allowance_bytes
+        ...
+        if post_footprint > self.hard_vram_limit_bytes:
+            raise VramBudgetExceededError(...)
+        self.total_cache_reclaims += 1
+        return result
```

---

## 7. CÁC RỦI RO CÒN LẠI (REMAINING RISKS)

1. **Tần suất gọi `empty_cache()`**: Nếu camera views liên tục nhảy vọt về độ phân giải, `empty_cache()` có thể được kích hoạt nhiều lần, gây suy giảm nhẹ tốc độ huấn luyện (~5 - 10 ms cho mỗi lần dọn cache). Tuy nhiên, trên tập `data/bonsai` (`images_4`), kích thước ảnh đồng nhất nên hiện tượng này chỉ xảy ra khi pool tích lũy đạt đỉnh.
2. **Ngưỡng Safety Headroom trong Mutation**: Tại chu kỳ mutation (bước 600, 700...), `evaluate_admission()` sử dụng `effective_budget_bytes = hard_budget - safety_headroom (512 MB)`. Nếu sau khi dọn cache mà bộ nhớ thực tế + transient cost của batch clone/split vượt $6,144 - 512 = 5,632$ MB, controller sẽ kích hoạt `PRUNE_FIRST` hoặc `SHRINK_AND_PRUNE_FIRST` đúng theo thiết kế thuật toán của PV-SRA-3DGS.

---

## 8. LỆNH CHẠY LẠI A3 SMOKE TEST (RETRY COMMAND)

Sử dụng thư mục output mới `output/smoke_test_bonsai_A3_retry1` (bảo toàn nguyên vẹn kết quả lần chạy trước tại `output/smoke_test_bonsai_A3`):

```powershell
python train.py `
  -s data/bonsai `
  -i images_4 `
  -m output/smoke_test_bonsai_A3_retry1 `
  --iterations 1000 `
  --save_iterations 1000 `
  --test_iterations 1000 `
  --densify_from_iter 500 `
  --densify_until_iter 15000 `
  --densification_interval 100 `
  --hard_vram_limit_mb 6144.0 `
  --pruning_mode decoupled `
  --occlusion_physics_mode corrected `
  --gradient_mask_mode canonical `
  --cost_calibration experiments/results/cost_calibration.json `
  --seed 0 `
  --eval
```

