# PV-SRA-3DGS Reproducibility Package: Configuration Reference

**Document Version:** 1.0  
**Project:** PV-SRA-3DGS  
**Repository:** ZeroGS  
**Date:** 2026-11-06  

---

### 1. Architectural Command-Line Arguments

The table below documents every configuration flag introduced by PV-SRA-3DGS, its default value, and its architectural significance:

| Argument Flag | Type | Default | Description & Mechanism |
|---|---|:---:|---|
| `--hard_vram_limit_mb` | float | `6144.0` | Maximum VRAM allocation target for PyTorch tensors (MiB). Excludes safety headroom and OS display allowance. |
| `--safety_headroom_mb` | float | `512.0` | Buffer reserved for unmanaged CUDA runtime allocations, cuBLAS contexts, and kernel fragmentation. |
| `--non_pytorch_allowance_mb` | float | `256.0` | Allowance for non-tensor allocations (e.g., driver metadata, display server spikes). |
| `--target_headroom_mb` | float | `1024.0` | Inflow admission target headroom used to dynamically scale candidate batch capacity $B_t$. |
| `--inflow_mode` | str | `dynamic` | Candidate admission mode: `dynamic` (headroom-proportional $B_t \in [B_{\min}, B_{\max}]$) or `static` (fixed $B_t$). |
| `--inflow_cap_min` | int | `5000` | Minimum per-iteration candidate admission floor $B_{\min}$ under dynamic inflow. |
| `--inflow_cap_max` | int | `15000` | Maximum per-iteration candidate admission ceiling $B_{\max}$ under dynamic inflow. |
| `--gaussians_per_iter_cap` | int | `5000` | Fixed per-iteration candidate admission limit when `--inflow_mode static` is active. |
| `--max_gaussians_cap` | int | `1150000` | Hard structural upper bound $N_{\max}$ on total admitted active Gaussians. |
| `--pruning_mode` | str | `legacy_as_is` | Primitive pruning lifecycle: `legacy_as_is` (preservation lifecycle, culls during macro resets) vs `decoupled` (periodic aggressive culling every densification step). |
| `--allocator_strategy` | str | `gain_per_byte` | Knapsack candidate selection: `gain_per_byte` (marginal utility greedy sorting) vs `quota`. |
| `--kappa_split` | float | `0.40` | Split candidate quota ceiling factor ($\kappa_{\text{split}} \in [0, 1]$), bounding the proportion of splits admitted in each mutation cycle. |
| `--tau_age` | int | `500` | Minimum age in training steps before a newly cloned Gaussian becomes eligible for opacity pruning. |
| `--tau_reset` | int | `400` | Cooldown period following a macro opacity reset during which opacity thresholds are suspended. |
| `--cost_calibration` | str | `experiments/results/cost_calibration.json` | Path to JSON file containing empirical affine cost model parameters ($c_{\text{clone}}, c_{\text{split}}$). |

---

### 2. Empirical Memory Cost Calibration Model

PV-SRA-3DGS models both persistent and transient memory consumption via calibrated linear coefficients:

#### 2.1 Persistent Footprint ($c_{\text{pers}}$)
Each active 3D Gaussian occupies exactly:
- **Geometry & Color Parameters:** $59 \times 4\text{ Bytes} = 236\text{ Bytes}$ ($\mathbf{x} \in \mathbb{R}^3$, $s \in \mathbb{R}^3$, $q \in \mathbb{R}^4$, $\alpha \in \mathbb{R}^1$, SH coefficients $\in \mathbb{R}^{48}$).
- **Adam Optimizer States:** Gradients ($236\text{ B}$) + 1st moment $\mathbf{m}$ ($236\text{ B}$) + 2nd moment $\mathbf{v}$ ($236\text{ B}$) = $708\text{ Bytes}$.
- **Total Persistent Cost:** $c_{\text{pers}} = 236 + 708 = \mathbf{944\text{ Bytes/Gaussian}}$.

#### 2.2 Transient Densification Footprint
During densification at iteration $t$, cloning and splitting create temporary tensor allocations for intermediate gradients, spatial sorting, and Adam state reallocation:
- **Clone Multiplier:** $\mu_{\text{clone}} = 1.204584 \implies c_{\text{clone}} = 944 \times 1.204584 = \mathbf{1{,}137.13\text{ Bytes}}$.
- **Split Multiplier:** $\mu_{\text{split}} = 2.573835 \implies c_{\text{split}} = 944 \times 2.573835 = \mathbf{2{,}429.70\text{ Bytes}}$.

The marginal knapsack allocator enforces the transient budget constraint:
$$\Delta M_{\text{transient}} = n_{\text{clone}} \cdot c_{\text{clone}} + n_{\text{split}} \cdot c_{\text{split}} \le B_{\text{transient}}(t)$$
with the split candidate quota:
$$n_{\text{split}} \le \kappa_{\text{split}} \cdot (n_{\text{clone}} + n_{\text{split}}), \quad \kappa_{\text{split}} = 0.40.$$

