import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import BEVDataset
from models import TrajectoryModel


def get_first(row, keys, default=None):
    for k in keys:
        if k in row:
            return row[k]
    return default


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def load_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt


def infer_one(model, sample, device):
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    with torch.no_grad():
        pred = model(batch)[0].detach().cpu().numpy()
    return pred


def draw_scene(ax, row, ref, gt, pred, title="", show_legend=False):
    # lane markings
    for e in row.get("lane_markings", []):
        pts = np.asarray(e.get("points", []), dtype=np.float32)
        if len(pts) == 0:
            continue
        c = e.get("color", "gray")
        if c == "white":
            c = "#d9bd84"
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2)

    # road boundaries
    for e in row.get("road_boundaries", []):
        pts = np.asarray(e.get("points", []), dtype=np.float32)
        if len(pts) == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="0.35", lw=2)

    ax.plot(ref[:, 0], ref[:, 1], "--", color="orange", lw=2.0, label="Biased reference")
    ax.plot(gt[:, 0], gt[:, 1], color="green", lw=2.6, label="GT path (80m)")
    ax.plot(pred[:, 0], pred[:, 1], color="red", lw=2.6, label="Predicted path")
    ax.scatter([0.0], [0.0], marker="^", s=90, color="blue", label="Ego")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("ego x (m)")
    ax.set_ylabel("ego y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.2)

    if show_legend:
        ax.legend(fontsize=9, loc="best")


def make_grid(records, output_path, title, ncols=3):
    if len(records) == 0:
        return

    n = len(records)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.reshape(-1)

    for j, rec in enumerate(records):
        ax = axes_flat[j]
        row = rec["row"]

        scene_id = rec["scene_id"]
        nav = rec["navigation"]
        dist = rec["ego_distance_m"]
        exit_lines = rec["observed_exit_lines"]
        mm = rec["model_mean_m"]
        me = rec["model_end_m"]
        rm = rec["ref_mean_m"]

        sub_title = (
            f"{scene_id} | {nav} | d={dist:.0f}m | exit_lines={exit_lines}\n"
            f"model_mean={mm:.2f}m end={me:.2f}m | ref_mean={rm:.2f}m"
        )

        draw_scene(
            ax,
            row=row,
            ref=rec["reference"],
            gt=rec["gt"],
            pred=rec["pred"],
            title=sub_title,
            show_legend=(j == 0),
        )

    for j in range(len(records), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_single_scene(rec, output_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    scene_id = rec["scene_id"]
    nav = rec["navigation"]
    dist = rec["ego_distance_m"]
    exit_lines = rec["observed_exit_lines"]
    mm = rec["model_mean_m"]
    me = rec["model_end_m"]
    rm = rec["ref_mean_m"]
    re = rec["ref_end_m"]

    title = (
        f"{scene_id} / {nav} / d={dist:.0f}m / exit_lines={exit_lines} | "
        f"model mean={mm:.2f}m end={me:.2f}m | "
        f"ref mean={rm:.2f}m end={re:.2f}m"
    )

    draw_scene(
        ax,
        row=rec["row"],
        ref=rec["reference"],
        gt=rec["gt"],
        pred=rec["pred"],
        title=title,
        show_legend=True,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evenly_pick(sorted_records, k):
    if len(sorted_records) <= k:
        return list(sorted_records)
    idxs = np.linspace(0, len(sorted_records) - 1, k, dtype=int).tolist()
    idxs = sorted(set(idxs))
    return [sorted_records[i] for i in idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v5_splits/test.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/best_path_v5.pt")
    parser.add_argument("--output-dir", default="outputs/v5_exit_visible_vis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-lines", type=int, default=32)
    parser.add_argument("--min-exit-lines", type=int, default=1,
                        help="只保留 observed_exit_lines >= 该值的样本")
    parser.add_argument("--distances", type=float, nargs="*", default=[20.0, 7.0],
                        help="只看这些 ego_distance_m；默认看 d20 和 d7")
    parser.add_argument("--per-nav", type=int, default=12,
                        help="每个导航类别挑多少张代表图（按误差从小到大均匀抽样）")
    parser.add_argument("--save-all", action="store_true",
                        help="是否把所有筛选后的样本逐张保存")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    horizon = int(ckpt.get("horizon", 81))

    ds = BEVDataset(
        args.data,
        split="all",
        augment=False,
        max_lines=args.max_lines,
        ref_points=horizon,
    )

    model = TrajectoryModel(horizon=horizon).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    distance_set = set(float(x) for x in args.distances) if args.distances else None

    records = []
    for idx in range(len(ds)):
        row = ds.rows[idx]
        sample = ds[idx]

        scene_id = str(get_first(row, ["id", "scene_id"], default=f"idx_{idx:04d}"))
        nav = str(get_first(row, ["navigation", "nav"], default="unknown"))

        meta = row.get("metadata", {})

        dist = to_float(
            meta.get("ego_distance_to_junction_m", -1)
        )

        observed_exit_lines = to_int(
            meta.get("observed_exit_lines", 0)
        )
        

        if observed_exit_lines < args.min_exit_lines:
            continue

        if distance_set is not None:
            # 用一个小容差处理 7 / 20 这类数
            keep = any(abs(dist - d) < 1e-6 or abs(dist - d) < 0.25 for d in distance_set)
            if not keep:
                continue

        ref = sample["reference"].cpu().numpy()
        gt = sample["target_xy"].cpu().numpy()
        pred = infer_one(model, sample, args.device)

        ref_err = np.linalg.norm(ref - gt, axis=1)
        model_err = np.linalg.norm(pred - gt, axis=1)

        records.append({
            "index": idx,
            "scene_id": scene_id,
            "navigation": nav,
            "ego_distance_m": dist,
            "observed_exit_lines": observed_exit_lines,
            "reference": ref,
            "gt": gt,
            "pred": pred,
            "ref_mean_m": float(ref_err.mean()),
            "ref_end_m": float(ref_err[-1]),
            "model_mean_m": float(model_err.mean()),
            "model_end_m": float(model_err[-1]),
            "improvement_m": float(ref_err.mean() - model_err.mean()),
            "row": row,
        })

    print(f"Filtered scenes: {len(records)}")
    if len(records) == 0:
        print("No scenes matched the filter.")
        return

    # 保存总表
    csv_path = out_dir / "selected_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "scene_id",
                "navigation",
                "ego_distance_m",
                "observed_exit_lines",
                "ref_mean_m",
                "ref_end_m",
                "model_mean_m",
                "model_end_m",
                "improvement_m",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "index": r["index"],
                "scene_id": r["scene_id"],
                "navigation": r["navigation"],
                "ego_distance_m": r["ego_distance_m"],
                "observed_exit_lines": r["observed_exit_lines"],
                "ref_mean_m": r["ref_mean_m"],
                "ref_end_m": r["ref_end_m"],
                "model_mean_m": r["model_mean_m"],
                "model_end_m": r["model_end_m"],
                "improvement_m": r["improvement_m"],
            })

    # overall grid
    overall_sorted = sorted(records, key=lambda x: x["model_mean_m"])
    overall_pick = evenly_pick(overall_sorted, min(len(overall_sorted), args.per_nav * 3))
    make_grid(
        overall_pick,
        out_dir / "all_selected_grid.png",
        title="V5 exit-visible cases (all navigations, sorted by model mean error)",
        ncols=3,
    )

    # per navigation
    navs = ["straight", "left", "right"]
    for nav in navs:
        nav_records = [r for r in records if r["navigation"] == nav]
        if len(nav_records) == 0:
            continue

        nav_sorted = sorted(nav_records, key=lambda x: x["model_mean_m"])
        nav_pick = evenly_pick(nav_sorted, min(args.per_nav, len(nav_sorted)))

        # grid
        make_grid(
            nav_pick,
            out_dir / f"{nav}_selected_grid.png",
            title=f"{nav}: exit-visible representative cases",
            ncols=3,
        )

        # single images
        single_dir = out_dir / nav / "selected"
        for rank, rec in enumerate(nav_pick):
            fname = (
                f"{rank:02d}_"
                f"{rec['scene_id']}_"
                f"d{int(round(rec['ego_distance_m']))}_"
                f"e{rec['observed_exit_lines']}_"
                f"mean{rec['model_mean_m']:.2f}.png"
            )
            save_single_scene(rec, single_dir / fname)

        # optional: save all
        if args.save_all:
            all_dir = out_dir / nav / "all"
            for rank, rec in enumerate(nav_sorted):
                fname = (
                    f"{rank:03d}_"
                    f"{rec['scene_id']}_"
                    f"d{int(round(rec['ego_distance_m']))}_"
                    f"e{rec['observed_exit_lines']}_"
                    f"mean{rec['model_mean_m']:.2f}.png"
                )
                save_single_scene(rec, all_dir / fname)

    print("Saved:")
    print(" ", csv_path)
    print(" ", out_dir / "all_selected_grid.png")
    for nav in navs:
        if (out_dir / f"{nav}_selected_grid.png").exists():
            print(" ", out_dir / f"{nav}_selected_grid.png")


if __name__ == "__main__":
    main()