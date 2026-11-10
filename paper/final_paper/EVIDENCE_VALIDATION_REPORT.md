# Automated Evidence Validation Report

- Raw physical runs inspected: 24/24
- Raw run/result/verification agreement: PASS
- Registry-to-raw agreement (PSNR, SSIM, LPIPS, peak VRAM): PASS
- Quantitative claims in the manuscript ledger: 26 verified, 0 unresolved
- Observed OOM events: 0/24
- Peak physical VRAM range: 6906.0-7656.0 MiB
- Paired Bonsai PSNR gain: 11.0850 +/- 2.3383 dB
- Paired t test: t(2)=8.2110, p=0.01451

The script treats files under `output/` as primary evidence. It does not alter raw evidence.
