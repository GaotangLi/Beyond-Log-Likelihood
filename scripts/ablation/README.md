# Ablation Scripts

This directory collects the paper ablation sweeps migrated from the developmental DeltaAI scripts:

- `convexity/figfont`: Qwen2.5-7B Figfont convexity sweep.
- `convexity/math`: Qwen2.5-Math-1.5B Numina-CoT convexity sweep.
- `model_scale`: Qwen2.5 model-scale sweep on math, with `original` and `p` objectives.
- `figure5`: thresholded percentile sweeps for `-p`, `-log(p)`, and `log(1-p)`.

Regenerate the shell scripts after editing the sweep definitions:

```bash
python scripts/ablation/generate_ablation_scripts.py
```

Run individual scripts from the repository root, for example:

```bash
bash scripts/ablation/model_scale/qwen-2.5-7b_p.sh
```
