# 7. Conclusion

In this paper, we addressed the fundamental challenge of training 3D Gaussian Splatting models under strict physical GPU memory constraints on consumer hardware. We identified the **Over-Pruning Dilemma**—a pervasive failure mode in bounded neural rendering where aggressive periodic opacity culling repeatedly destroys developing fine geometry, inflicting catastrophic visual degradation exceeding $11\text{ dB}$.

To overcome this limitation, we introduced **PV-SRA-3DGS** (Peak-VRAM-Constrained Spatial Resource Allocation), a principled framework that replaces destructive culling with proactive admission control and developmental preservation. By integrating an analytical affine cost model ($944\text{ bytes}$ persistent, $1{,}137\text{ bytes}$ clone, $2{,}429\text{ bytes}$ split), a Dynamic Headroom-Aware Inflow Governor, a Marginal Knapsack Spatial Allocator with explicit split quotas ($\kappa_{\text{split}} = 0.40$), and a Preservation-Centric Lifecycle, our architecture enables stable, high-fidelity neural radiance reconstruction strictly within an 8GB VRAM envelope.

Extensive physical hardware benchmarks on an NVIDIA GeForce RTX 5060 Laptop GPU across 23 full 30,000-iteration trials established:
1. **Zero Out-Of-Memory exceptions** across all executed experiments, maintaining peak device memory strictly within $6{,}906\text{--}7{,}570\text{ MiB}$.
2. **State-of-the-art reconstruction fidelity on `bonsai` ($31.83 \pm 0.48\text{ dB}$ PSNR)**, establishing a statistically significant **$+11.0850 \pm 2.3383\text{ dB}$ paired gain** over baseline bounded models ($p = 0.01451$, Cohen's $d = 4.7406$) and surpassing unconstrained Vanilla 3DGS ($29.59\text{ dB}$) by **$+2.24\text{ dB}$** with **$40.2\%$ fewer primitives**.
3. **Robust cross-scene generalizability**, delivering a **$+6.2621\text{ dB}$ gain on `garden`** and a **$+2.9422\text{ dB}$ gain on `flowers`** under extreme spatial candidate generation ($8.91\text{M}$ requests).

By proving that spatial neural representations can be trained safely and with superior fidelity under strict memory boundaries, PV-SRA-3DGS provides an essential step toward democratizing high-fidelity radiance field optimization on consumer workstations, edge devices, and mobile platforms.

