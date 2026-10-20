"""V4 synthetic BEV geometry -> fixed-size tensors for time-free path learning."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

NAV = {'straight': 0, 'left': 1, 'right': 2}


def resample(points, n, *, step=None):
    """Resample an ordered polyline by arc length; robust to repeated points."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros((n, 2), np.float32)
    if len(p) == 1:
        return np.repeat(p, n, axis=0)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(d) > 1e-5]
    p, d = p[keep], d[keep]
    if len(p) == 1:
        return np.repeat(p, n, axis=0)
    # For the reference, use distance-from-start samples, not time-indexed samples.
    t = np.linspace(0.0, float(d[-1]), n) if step is None else np.minimum(np.arange(n) * step, d[-1])
    return np.stack([np.interp(t, d, p[:, j]) for j in range(2)], -1).astype(np.float32)


class BEVDataset(Dataset):
    def __init__(self, path, split='train', seed=42, val_fraction=0.2,
                 max_lines=16, line_points=32, ref_points=None, augment=False):
        with Path(path).open(encoding='utf-8') as f:
            rows = [json.loads(s) for s in f if s.strip()]
        if not rows:
            raise ValueError('Dataset is empty: ' + str(path))
        if 'gt_path' not in rows[0]:
            raise ValueError('Expected V4 gt_path, but this appears to be an old trajectory dataset')
        self.path_points = len(rows[0]['gt_path'])
        self.spacing = float(rows[0].get('gt_path_sample_spacing_m', 1.0))
        self.ref_points = int(ref_points or self.path_points)
        self.max_lines, self.line_points = max_lines, line_points
        self.augment = augment and split == 'train'
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        nv = max(1, round(len(rows) * val_fraction)) if len(rows) > 1 else 0
        if split == 'train':
            indices = indices[nv:]
        elif split == 'val':
            indices = indices[:nv]
        elif split != 'all':
            raise ValueError("split must be 'train', 'val' or 'all'")
        self.rows = [rows[i] for i in indices]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        lines = np.zeros((self.max_lines, self.line_points, 11), dtype=np.float32)
        valid = np.zeros(self.max_lines, dtype=bool)
        elems = [(e, False) for e in row['lane_markings']] + [(e, True) for e in row['road_boundaries']]
        # Permute polylines during training (set order has no physical meaning).
        if self.augment:
            random.shuffle(elems)
        for i, (e, boundary) in enumerate(elems[:self.max_lines]):
            pts = resample(e['points'], self.line_points)
            if self.augment and random.random() < 0.5:
                pts = pts[::-1].copy()
            delta = np.gradient(pts, axis=0)
            lines[i, :, :2] = pts
            lines[i, :, 2:4] = delta
            lines[i, :, 4] = float(e.get('confidence', 1.0))
            lines[i, :, 5] = float(not boundary)
            lines[i, :, 6] = float(boundary)
            lines[i, :, 7] = float(e.get('color') == 'white')
            lines[i, :, 8] = float(e.get('color') == 'yellow')
            lines[i, :, 9] = float(e.get('line_style') == 'solid')
            lines[i, :, 10] = float(e.get('line_style') == 'dashed')
            valid[i] = True
        reference = resample(row['reference_line'], self.ref_points, step=self.spacing)
        # Translate the biased reference's first sample to ego origin; preserve shape/bias elsewhere.
        reference = reference - reference[:1]
        gt = np.asarray(row['gt_path'], dtype=np.float32)
        if gt.shape != (self.path_points, 2):
            raise ValueError(f"Inconsistent gt_path shape: {row['id']} {gt.shape}")
        return {
            'lines': torch.from_numpy(lines),
            'line_valid': torch.from_numpy(valid),
            'reference': torch.from_numpy(reference.copy()),
            'navigation': torch.tensor(NAV[row['navigation']], dtype=torch.long),
            'target_xy': torch.from_numpy(gt.copy()),
        }
