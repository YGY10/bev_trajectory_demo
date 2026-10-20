# Synthetic BEV Path V4

This is an 80-meter arc-length-sampled **path**, not a time-indexed trajectory.

## Conventions
- Ego frame: x forward, y left. BEV FOV uses symmetric x limits `[-xlim, +xlim]`.
- Straight: nominal x ±80 m, y ±10 m, with 80% probability of a nearby limit; otherwise shorter FOV.
- Left/right: nominal x ±60 m, y ±20 m, with 80% probability of a nearby limit; otherwise shorter FOV.
- Visible polylines get small correlated coordinate perturbations and occasional contiguous gaps (`--bev-noise medium`). Boundary corruption is less severe than lane-marking corruption.
- `gt_path` has 81 points at ~1 m arc-length spacing (0 through 80 m); GT and `reference_line` are not perturbed by BEV noise. Reference-line map bias is generated separately.
- Input line budget is <=16 elements for compatibility with earlier demo's polyline encoder.

## Generate
```bash
python data/generate_dataset_v4.py --count 3000 --bev-noise medium \
  --path-length 80 --spacing 1 \
  --output data/synthetic_bev_paths_v4.jsonl \
  --preview data/sample_scenes_v4.png
```
`--bev-noise` choices: `none`, `mild`, `medium`, `strong`.

## WARNING
Earlier trajectory-based `train.py`, `evaluate.py`, and `bev_dataset.py` are not compatible with `gt_path`; update training to the time-free 81-point path representation first.
