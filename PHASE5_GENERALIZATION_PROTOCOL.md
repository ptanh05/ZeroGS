# Phase 5 Experimental Protocol: Cross-Scene Generalization Study

**Document Target:** Protocol specification for expanding ZeroGS 4GB validation across diverse scene topologies on the Mip-NeRF 360 benchmark.  
**Target Scenes:** `bicycle`, `bonsai`, `flowers`, `stump`  
**Enforced Budget:** Hard 4,096 MiB Physical VRAM Limit  
**Hardware Target:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB Physical VRAM)  
**Standard:** Strict Experimental Design (Do NOT execute until authorized)  

---

## 1. Scientific Objective

To prove that ZeroGS is not overfitted to `garden`, this protocol evaluates the system across four representative scene topologies spanning indoor bounded environments, thin outdoor structures, delicate biological geometry, and natural woodland surfaces.

---

## 2. Scene Topology & Complexity Characterization

| Scene | Environment Type | Image Directory | Resolution | Cameras (Train / Test) | Architectural Challenge |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`bicycle`** | Outdoor Unbounded | `images_4` | $1238 \times 824$ | $194$ ($170$ / $24$) | High-frequency bicycle spokes, chain links, background foliage, wide depth range. |
| **`bonsai`** | Indoor Bounded | `images_4` | $952 \times 632$ | $292$ ($255$ / $37$) | High camera count ($292$), intricate bonsai needle clusters, complex wooden table specularity. |
| **`flowers`** | Outdoor Intricate | `images_4` | $1297 \times 840$ | $173$ ($151$ / $22$) | Very thin flower petals, complex depth-of-field blur, highly susceptible to Over-Pruning Dilemma. |
| **`stump`** | Outdoor Natural | `images_4` | $1238 \times 824$ | $200$ ($175$ / $25$) | Multi-scale bark textures, uneven woodland ground, high geometric irregularity. |

---

## 3. Dataset-Specific Memory Limits & Wall Analysis

### 3.1 Static Camera Memory (Wall 1 Risk)
Without `--data_device cpu`, each scene locks an unsustainable portion of the 4,096 MiB budget purely in static images:
- **`bonsai`:** $292 \times (952 \times 632 \times 3 \times 4\text{ B}) \approx \mathbf{2{,}109\text{ MiB}}$ ($51.5\%$ of 4GB budget).
- **`bicycle`:** $194 \times (1238 \times 824 \times 3 \times 4\text{ B}) \approx \mathbf{2{,}372\text{ MiB}}$ ($57.9\%$ of 4GB budget).
- **`flowers`:** $173 \times (1297 \times 840 \times 3 \times 4\text{ B}) \approx \mathbf{3{,}015\text{ MiB}}$ ($73.6\%$ of 4GB budget).
- **`stump`:** $200 \times (1238 \times 824 \times 3 \times 4\text{ B}) \approx \mathbf{2{,}445\text{ MiB}}$ ($59.7\%$ of 4GB budget).

> [!CRITICAL]
> **Enforcement Rule:** Every experiment must activate `--data_device cpu` (Pageable Host Streaming) to reduce GPU camera footprint to $\le 42\text{ MB}$, leaving $\approx 3{,}800\text{ MB}$ free for model optimization.

---

## 4. Standardized Execution Configuration

All four scenes will be evaluated using strictly identical parameters:
- **Seed:** `--seed 0`
- **Total Iterations:** $30{,}000$ steps
- **Densification Schedule:** Start step $1{,}000$, Stop step $15{,}000$, Interval $100$ steps
- **Opacity Reset:** Macro reset every $3{,}000$ steps with $\tau_{\text{reset}} = 400$
- **Primitive Preservation Window:** $\tau_{\text{age}} = 500$
- **Split Ceiling:** $\kappa_{\text{split}} = 0.40$
- **Admission Inflow:** Dynamic $[5\text{k}, 15\text{k}]$ based on target headroom $1{,}024\text{ MiB}$
- **Optimizer:** `Int8GaussianAdam` (Logarithmic uint8 dynamic range)
- **Hard VRAM Cap:** $4{,}096.0\text{ MiB}$ (PyTorch effective ceiling $3{,}584.0\text{ MiB}$)

### Standard Command Template:
```bash
python scripts/run_generalization_scene.py \
    --source_path data/<scene> \
    --images images_4 \
    --output_dir experiments/results/generalization_4gb/<scene>_zerogs_int8 \
    --data_device cpu \
    --memory_budget_mb 4096.0 \
    --optimizer_type int8 \
    --iterations 30000 \
    --save_iterations 7000 15000 30000 \
    --test_iterations 7000 15000 30000
```

---

## 5. Success Criteria & Publication Target Metrics

| Scene | Target 30k PSNR | Target 30k SSIM | Target 30k LPIPS | Expected Primitives | Peak Reserved VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bicycle`** | $\ge 22.0\text{ dB}$ | $\ge 0.600$ | $\le 0.420$ | $450\text{k}\text{--}600\text{k}$ | $\le 3{,}600\text{ MB}$ |
| **`bonsai`** | $\ge 29.0\text{ dB}$ | $\ge 0.890$ | $\le 0.250$ | $350\text{k}\text{--}450\text{k}$ | $\le 3{,}700\text{ MB}$ |
| **`flowers`** | $\ge 19.5\text{ dB}$ | $\ge 0.520$ | $\le 0.450$ | $400\text{k}\text{--}550\text{k}$ | $\le 3{,}500\text{ MB}$ |
| **`stump`** | $\ge 23.0\text{ dB}$ | $\ge 0.650$ | $\le 0.380$ | $450\text{k}\text{--}600\text{k}$ | $\le 3{,}600\text{ MB}$ |

### Zero-Tolerance Scientific Gate:
1. Every run must finish with `exit_code == 0`.
2. Telemetry must confirm `Peak Reserved VRAM <= 4096.0 MB` across all cycles.
3. Checkpoint hashes and test renders must pass automated NaN/Inf inspection.
