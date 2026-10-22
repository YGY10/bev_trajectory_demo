import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def flag(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def finite_values(rows, key):
    return [
        number(r, key)
        for r in rows
        if math.isfinite(number(r, key))
    ]


def avg(rows, key):
    values = finite_values(rows, key)
    return mean(values) if values else math.nan


def med(rows, key):
    values = finite_values(rows, key)
    return median(values) if values else math.nan


def load_visibility(data_path):
    visibility = {}
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            n = int(meta.get("observed_exit_lines", 0))
            visibility[row["id"]] = {
                "observed_exit_lines": n,
                "exit_visibility": "visible" if n > 0 else "none",
                "observed_entry_lines": int(
                    meta.get("observed_entry_lines", 0)
                ),
            }
    return visibility


def select_original(path):
    # Compare the actual model prediction (alpha=1), not
    # the reference baseline or the duplicate nav_scaled row.
    rows = [
        r for r in read_csv(path)
        if r.get("method") == "original"
        and abs(number(r, "alpha") - 1.0) < 1e-6
    ]
    result = {}
    for row in rows:
        sid = row["scene_id"]
        if sid in result:
            raise ValueError(f"Duplicate original prediction: {sid}")
        result[sid] = row
    return result


def write_csv(path, rows):
    if not rows:
        print("No rows to save:", path)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(group_type, group_name, rows, ade_thr, exit_thr):
    eligible = [
        r for r in rows
        if r["exit_eligible_d1"] and r["exit_eligible_refaug"]
        and math.isfinite(r["delta_exit_m"])
    ]

    return {
        "group_type": group_type,
        "group": group_name,
        "n_scenes": len(rows),
        "n_exit_eligible": len(eligible),
        "mean_delta_ade_m": avg(rows, "delta_ade_m"),
        "median_delta_ade_m": med(rows, "delta_ade_m"),
        "mean_delta_fde_m": avg(rows, "delta_fde_m"),
        "mean_delta_exit_m": avg(eligible, "delta_exit_m"),
        "median_delta_exit_m": med(eligible, "delta_exit_m"),
        "exit_worsened_count": sum(
            r["delta_exit_m"] > exit_thr for r in eligible
        ),
        "exit_improved_count": sum(
            r["delta_exit_m"] < -exit_thr for r in eligible
        ),
        "ade_improved_exit_worsened_count": sum(
            r["delta_ade_m"] < -ade_thr
            and r["delta_exit_m"] > exit_thr
            for r in eligible
        ),
        "center_hit_dropped_count": sum(
            r["delta_center_hit"] < -1e-6
            for r in eligible
        ),
        "lane_hit_dropped_count": sum(
            r["delta_lane_hit"] < -1e-6
            for r in eligible
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="outputs/v6B1_D1_best_ade_nav/per_scene.csv",
    )
    parser.add_argument(
        "--candidate",
        default="outputs/v6B1_RefAug_v2_best_ade_nav/per_scene.csv",
    )
    parser.add_argument(
        "--data", default="data/v5_splits/val.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v6B1_RefAug_v2_regression",
    )
    parser.add_argument(
        "--ade-threshold", type=float, default=0.01,
        help="ADE improvement threshold in meters",
    )
    parser.add_argument(
        "--exit-threshold", type=float, default=0.05,
        help="Exit regression threshold in meters",
    )
    args = parser.parse_args()

    d1 = select_original(args.baseline)
    refaug = select_original(args.candidate)
    visibility = load_visibility(args.data)

    common = sorted(set(d1) & set(refaug))
    if not common:
        raise ValueError("No matching scene_id values")

    records = []
    for sid in common:
        a, b = d1[sid], refaug[sid]
        vis = visibility.get(sid, {})

        da = number(b, "path_ade_m") - number(a, "path_ade_m")
        df = number(b, "path_fde_m") - number(a, "path_fde_m")

        ea, eb = flag(a["exit_eligible"]), flag(b["exit_eligible"])
        both_eligible = ea and eb

        delta_exit = (
            number(b, "exit_lat_m") - number(a, "exit_lat_m")
            if both_eligible else math.nan
        )
        delta_improvement = (
            number(b, "exit_improvement_vs_ref_m")
            - number(a, "exit_improvement_vs_ref_m")
            if both_eligible else math.nan
        )

        ca, cb = number(a, "center_hit"), number(b, "center_hit")
        la, lb = number(a, "lane_hit"), number(b, "lane_hit")

        rec = {
            "scene_id": sid,
            "base_intersection_id": a["base_intersection_id"],
            "navigation": a["navigation"],
            "ego_distance_m": number(a, "ego_distance_m"),
            "exit_visibility": vis.get("exit_visibility", "unknown"),
            "observed_exit_lines": vis.get("observed_exit_lines", ""),
            "observed_entry_lines": vis.get("observed_entry_lines", ""),
            "visible_target_points": number(a, "visible_target_points"),
            "exit_eligible_d1": ea,
            "exit_eligible_refaug": eb,
            "d1_ade_m": number(a, "path_ade_m"),
            "refaug_ade_m": number(b, "path_ade_m"),
            "delta_ade_m": da,
            "d1_fde_m": number(a, "path_fde_m"),
            "refaug_fde_m": number(b, "path_fde_m"),
            "delta_fde_m": df,
            "d1_exit_m": number(a, "exit_lat_m") if ea else math.nan,
            "refaug_exit_m": number(b, "exit_lat_m") if eb else math.nan,
            "delta_exit_m": delta_exit,
            "d1_exit_improvement_vs_ref_m": (
                number(a, "exit_improvement_vs_ref_m") if ea else math.nan
            ),
            "refaug_exit_improvement_vs_ref_m": (
                number(b, "exit_improvement_vs_ref_m") if eb else math.nan
            ),
            "delta_exit_improvement_m": delta_improvement,
            "d1_center_hit": ca,
            "refaug_center_hit": cb,
            "delta_center_hit": cb - ca,
            "d1_lane_hit": la,
            "refaug_lane_hit": lb,
            "delta_lane_hit": lb - la,
        }

        rec["ade_improved"] = da < -args.ade_threshold
        rec["exit_worsened"] = (
            both_eligible and delta_exit > args.exit_threshold
        )
        rec["ade_better_exit_worse"] = (
            rec["ade_improved"] and rec["exit_worsened"]
        )
        records.append(rec)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "per_scene_comparison.csv", records)

    # Summarize overall, by navigation, distance, and actual
    # exit-marking visibility metadata from the source dataset.
    groups = [("all", "all", records)]
    for field, label in [
        ("navigation", "navigation"),
        ("ego_distance_m", "distance_m"),
        ("exit_visibility", "exit_visibility"),
    ]:
        values = sorted({str(r[field]) for r in records})
        for value in values:
            subset = [r for r in records if str(r[field]) == value]
            groups.append((label, value, subset))

    summaries = [
        summarize(gt, name, rows, args.ade_threshold, args.exit_threshold)
        for gt, name, rows in groups
    ]
    write_csv(out / "summary.csv", summaries)

    # Rank eligible scenes by largest increase in exit error.
    eligible = [
        r for r in records
        if r["exit_eligible_d1"] and r["exit_eligible_refaug"]
    ]
    worst = sorted(
        eligible, key=lambda r: r["delta_exit_m"], reverse=True
    )
    write_csv(out / "worst_exit_regressions.csv", worst[:50])

    mixed = [r for r in eligible if r["ade_better_exit_worse"]]
    write_csv(out / "ade_better_exit_worse.csv", mixed)

    print(f"Matched scenes: {len(records)}")
    print(f"Exit-eligible in both: {len(eligible)}")
    print(
        "ADE improved but exit worsened:",
        f"{len(mixed)}/{len(eligible)}",
        f"({100 * len(mixed) / max(1, len(eligible)):.1f}%)",
    )
    print("\n===== Group summaries =====")
    for r in summaries:
        print(
            f'{r["group_type"]:15s} {r["group"]:12s} '
            f'n={r["n_scenes"]:3d} '
            f'dADE={r["mean_delta_ade_m"]:+.4f}m '
            f'dExit={r["mean_delta_exit_m"]:+.4f}m '
            f'mixed={r["ade_improved_exit_worsened_count"]}'
        )

    print("\n===== Worst exit regressions =====")
    for r in worst[:15]:
        print(
            f'{r["scene_id"]:40s} '
            f'{r["navigation"]:8s} '
            f'd={r["ego_distance_m"]:4.0f}m '
            f'visible={r["exit_visibility"]:7s} '
            f'ADE {r["delta_ade_m"]:+.3f}m '
            f'exit {r["delta_exit_m"]:+.3f}m '
            f'center {r["d1_center_hit"]:.0f}'
            f'->{r["refaug_center_hit"]:.0f}'
        )

    print("\nSaved results to:", out)


if __name__ == "__main__":
    main()
