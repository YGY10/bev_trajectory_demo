# V5 intersection data — corrected exit-lane visibility

The previous V5 generator clipped all turn scene lines to ego-frame `y ∈ [-20,20] m`, while the exit road starts at |y| ≈ 20–24 m. This mistakenly removed most outgoing lane lines.

This revision retains the **entrance BEV core** (`x ±60 m`, `y ±20 m`) and adds a **turn-oriented visible corridor** for the outgoing road, with a randomized visible extent of 22–35 m along the exit. It is **not** a strict fixed ego-axis rectangle or a physically calibrated camera visibility model. If your actual sensor is strictly limited to |y|≤20 m, these farther exit lines cannot be present: you must instead place the corner closer or use true sensor geometry.

The intersection interior has no painted connecting lane lines. The available approach/exit lines have small position noise, optional gaps, and variable length; labels and reference/GT remain unchanged. V4 JSONL schema retained.

Generated scenes: 9 (3 directions × 3 lane pairings). Output remains an 81-point 80 m path.

## Commands

```bash
conda activate toybev
python data/generate_intersections.py \
  --output data/intersection_v5_fixed.jsonl \
  --preview outputs/intersection_v5_fixed_preview.png

python -u evaluate_zero_shot.py \
  --data data/intersection_v5_fixed.jsonl \
  --checkpoint checkpoints/best_path_v4.pt \
  --output-dir outputs/intersection_v5_fixed_zeroshot
```

Run from project root. Copy `generate_intersections.py` into `data/` and `evaluate_zero_shot.py` into project root, if not already there.

The metadata `bev_box_m` describes the ego-frame core, not the union with the turn-oriented corridor. `turn_exit_corridor_length_m` records that extra corridor's length. This synthetic fixture is for transfer testing, not proof of human-like path generation.
