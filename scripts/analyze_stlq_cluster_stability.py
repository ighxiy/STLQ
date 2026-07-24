#!/usr/bin/env python3
"""Compare per-cluster STLQ hit-rate stability across runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GRAPH_METRICS = [
    "cluster_mean_depth",
    "cluster_degree_gini",
    "cluster_linked_ratio",
    "cluster_n_virtual",
    "cluster_max_depth",
    "cluster_max_child_degree",
    "cluster_max_subtree_real",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graph_hit_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_queries", type=int, default=20)
    return p.parse_args()


def weighted_corr(xs: list[float], ys: list[float], ws: list[float]) -> float:
    sw = sum(ws)
    if sw <= 0:
        return 0.0
    mx = sum(x * w for x, w in zip(xs, ws)) / sw
    my = sum(y * w for y, w in zip(ys, ws)) / sw
    cov = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sw
    vx = sum(w * (x - mx) * (x - mx) for x, w in zip(xs, ws)) / sw
    vy = sum(w * (y - my) * (y - my) for y, w in zip(ys, ws)) / sw
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.graph_hit_json).read_text())
    names = list(data)

    cluster_by_run: dict[str, dict[int, dict]] = {}
    for name, run in data.items():
        rows = {}
        for cid_s, row in run["gt_cluster_stats"].items():
            if row["n"] >= args.min_queries:
                rows[int(cid_s)] = row
        cluster_by_run[name] = rows

    pair_corr = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sorted(set(cluster_by_run[a]) & set(cluster_by_run[b]))
            xs = [cluster_by_run[a][cid]["hit_rate"] for cid in common]
            ys = [cluster_by_run[b][cid]["hit_rate"] for cid in common]
            ws = [min(cluster_by_run[a][cid]["n"], cluster_by_run[b][cid]["n"]) for cid in common]
            pair_corr.append(
                {
                    "run_a": a,
                    "run_b": b,
                    "common_clusters": len(common),
                    "weighted_corr_hit_rate": weighted_corr(xs, ys, ws),
                }
            )

    cluster_union = sorted(set().union(*(set(v) for v in cluster_by_run.values())))
    cluster_summary = []
    for cid in cluster_union:
        vals = []
        ns = []
        metrics = {m: [] for m in GRAPH_METRICS}
        present = []
        for name in names:
            row = cluster_by_run[name].get(cid)
            if row is None:
                continue
            vals.append(row["hit_rate"])
            ns.append(row["n"])
            present.append(name)
            for m in GRAPH_METRICS:
                metrics[m].append(row[m])
        if len(vals) < 2:
            continue
        avg = sum(vals) / len(vals)
        var = sum((x - avg) ** 2 for x in vals) / len(vals)
        cluster_summary.append(
            {
                "cid": cid,
                "runs": len(vals),
                "total_gt_queries": sum(ns),
                "avg_hit_rate": avg,
                "std_hit_rate": math.sqrt(var),
                "present": present,
                **{m: sum(v) / len(v) for m, v in metrics.items()},
            }
        )

    stable = [r for r in cluster_summary if r["runs"] >= 3]
    stable_good = sorted(stable, key=lambda r: (-r["avg_hit_rate"], -r["runs"], -r["total_gt_queries"]))[:15]
    stable_bad = sorted(stable, key=lambda r: (r["avg_hit_rate"], -r["runs"], -r["total_gt_queries"]))[:15]
    volatile = sorted(stable, key=lambda r: (-r["std_hit_rate"], -r["total_gt_queries"]))[:15]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "min_queries": args.min_queries,
        "run_names": names,
        "pairwise_correlations": pair_corr,
        "stable_good": stable_good,
        "stable_bad": stable_bad,
        "volatile": volatile,
    }
    (out_dir / "cluster_stability.json").write_text(json.dumps(result, indent=2))

    lines = []
    lines.append("# STLQ Cluster Stability Analysis")
    lines.append("")
    lines.append(f"Per-cluster GT1 hit rates are compared across runs. Clusters with fewer than {args.min_queries} GT1-in-shortlist queries in a run are ignored for that run.")
    lines.append("")
    lines.append("## Pairwise Hit-Rate Correlation")
    lines.append("")
    lines.append("| run A | run B | common clusters | weighted corr(hit_rate) |")
    lines.append("|---|---|---:|---:|")
    for r in pair_corr:
        lines.append(f"| {r['run_a']} | {r['run_b']} | {r['common_clusters']} | {r['weighted_corr_hit_rate']:.4f} |")
    lines.append("")

    def table(title: str, rows: list[dict]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| cid | runs | total_gt_queries | avg_hit_rate | std_hit_rate | mean_depth | degree_gini | linked_ratio | n_virtual |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['cid']} | {r['runs']} | {r['total_gt_queries']} | {r['avg_hit_rate']:.4f} | {r['std_hit_rate']:.4f} | {r['cluster_mean_depth']:.3f} | {r['cluster_degree_gini']:.3f} | {r['cluster_linked_ratio']:.4f} | {r['cluster_n_virtual']:.0f} |"
            )
        lines.append("")

    table("Stable Good Clusters", stable_good)
    table("Stable Bad Clusters", stable_bad)
    table("High-Variance Clusters", volatile)
    (out_dir / "cluster_stability.md").write_text("\n".join(lines) + "\n")
    print(f"[cluster-stability] wrote {out_dir / 'cluster_stability.md'}")


if __name__ == "__main__":
    main()
