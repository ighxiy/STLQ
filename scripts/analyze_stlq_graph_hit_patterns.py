#!/usr/bin/env python3
"""Analyze STLQ hit/miss patterns against linkage graph features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


PARENT_KINDS = ["root", "super_root", "virtual", "real_prev_depth", "unknown"]
REGIMES = ["with_virtual", "no_virtual_deep", "no_virtual_mid", "no_virtual_shallow"]
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
    p.add_argument("--csv", action="append", required=True)
    p.add_argument("--name", action="append")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def parent_kind(row: dict[str, str]) -> str:
    k = row.get("parent_kind", "unknown")
    return k if k in PARENT_KINDS else "unknown"


def regime(row: dict[str, str]) -> str:
    if f(row, "cluster_n_virtual") > 0:
        return "with_virtual"
    md = f(row, "cluster_mean_depth")
    if md >= 3.0:
        return "no_virtual_deep"
    if md >= 2.0:
        return "no_virtual_mid"
    return "no_virtual_shallow"


def depth_bin(row: dict[str, str]) -> str:
    d = int(round(f(row, "depth", 0.0)))
    if d <= 0:
        return "d0"
    if d == 1:
        return "d1"
    if d == 2:
        return "d2"
    if d <= 4:
        return "d3_4"
    return "d5p"


def bucket(row: dict[str, str]) -> tuple[str, str, str]:
    return (parent_kind(row), regime(row), depth_bin(row))


def add_stat(d: dict, key, hit: bool, values: dict[str, float]) -> None:
    s = d.setdefault(key, {"n": 0, "hit": 0, "values": defaultdict(float)})
    s["n"] += 1
    s["hit"] += int(hit)
    for k, v in values.items():
        s["values"][k] += v


def finalize_stat(d: dict) -> dict:
    out = {}
    for key, s in d.items():
        n = s["n"]
        vals = {k: v / n for k, v in s["values"].items()}
        out["|".join(key) if isinstance(key, tuple) else str(key)] = {
            "n": n,
            "hit_rate": s["hit"] / n if n else 0.0,
            **vals,
        }
    return out


def weighted_corr(xs: list[float], ys: list[float], ws: list[float]) -> float:
    sw = sum(ws)
    if sw <= 0:
        return 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    cov = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sw
    vx = sum(w * (x - mx) * (x - mx) for x, w in zip(xs, ws)) / sw
    vy = sum(w * (y - my) * (y - my) for y, w in zip(ys, ws)) / sw
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def finalize_cluster_stats(cluster_stats: dict) -> tuple[dict, dict]:
    rows = []
    for cid, s in cluster_stats.items():
        n = s["n"]
        if n <= 0:
            continue
        row = {
            "cid": cid,
            "n": n,
            "hit_rate": s["hit"] / n,
            "gt_adc_rank_mean": s["gt_adc_rank"] / n,
            "adc_margin_mean": s["adc_margin"] / n,
        }
        for m in GRAPH_METRICS:
            row[m] = s[m] / n
        rows.append(row)

    eligible = [r for r in rows if r["n"] >= 20]
    corr = {}
    for target in ["hit_rate", "gt_adc_rank_mean", "adc_margin_mean"]:
        corr[target] = {}
        for m in GRAPH_METRICS:
            corr[target][m] = weighted_corr(
                [r[m] for r in eligible],
                [r[target] for r in eligible],
                [r["n"] for r in eligible],
            )

    top_good = sorted(eligible, key=lambda r: (-r["hit_rate"], -r["n"]))[:10]
    top_bad = sorted(eligible, key=lambda r: (r["hit_rate"], -r["n"]))[:10]
    compact_rows = {
        str(r["cid"]): {
            k: r[k]
            for k in [
                "n",
                "hit_rate",
                "gt_adc_rank_mean",
                "adc_margin_mean",
                *GRAPH_METRICS,
            ]
        }
        for r in rows
    }
    return compact_rows, {
        "eligible_clusters_n_ge_20": len(eligible),
        "correlations": corr,
        "top_good_clusters_n_ge_20": top_good,
        "top_bad_clusters_n_ge_20": top_bad,
    }


def quantile_edges(values: list[float], bins: int = 4) -> list[float]:
    if not values:
        return []
    xs = sorted(values)
    edges = []
    for i in range(1, bins):
        pos = int(round(i * (len(xs) - 1) / bins))
        edges.append(xs[pos])
    return edges


def quantile_label(v: float, edges: list[float]) -> str:
    lo = "-inf"
    for i, e in enumerate(edges):
        if v <= e:
            return f"q{i+1}:{lo}..{e:.6g}"
        lo = f"{e:.6g}"
    return f"q{len(edges)+1}:{lo}..inf"


def load_query_records(csv_path: Path) -> list[dict]:
    records: list[dict] = []
    cur_qid = None
    rows: list[dict[str, str]] = []

    def flush() -> None:
        if not rows:
            return
        top = min(rows, key=lambda r: int(r["rank"]))
        gt = next((r for r in rows if int(r.get("is_gt1", "0") or "0") != 0), None)
        hit = gt is not None and int(top.get("is_gt1", "0") or "0") != 0
        rec = {
            "qid": int(top["query_id"]),
            "hit": hit,
            "gt_in_shortlist": gt is not None,
            "top": top,
            "gt": gt,
        }
        records.append(rec)

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            qid = int(row["query_id"])
            if cur_qid is None:
                cur_qid = qid
            if qid != cur_qid:
                flush()
                rows = []
                cur_qid = qid
            rows.append(row)
    flush()
    return records


def analyze_records(records: list[dict]) -> dict:
    n = len(records)
    hits = sum(1 for r in records if r["hit"])
    gt_in = sum(1 for r in records if r["gt_in_shortlist"])

    top_bucket_stats = {}
    gt_bucket_stats = {}
    top_parent = {}
    gt_parent = {}
    top_regime = {}
    gt_regime = {}
    miss_parent_pair = defaultdict(int)
    miss_regime_pair = defaultdict(int)
    gt_cluster_stats = defaultdict(lambda: defaultdict(float))

    gt_metric_values = {m: [] for m in GRAPH_METRICS}
    for r in records:
        if r["gt"] is not None:
            for m in GRAPH_METRICS:
                gt_metric_values[m].append(f(r["gt"], m))
    metric_edges = {m: quantile_edges(vals) for m, vals in gt_metric_values.items()}
    metric_bins = {m: {} for m in GRAPH_METRICS}

    miss_margins = []
    hit_margins = []
    miss_by_top_exact_rank = defaultdict(int)

    for r in records:
        top = r["top"]
        gt = r["gt"]
        hit = bool(r["hit"])
        top_values = {
            "exact_rank_mean": f(top, "exact_rank_in_shortlist"),
            "recon_minus_exact_mean": f(top, "recon_minus_exact"),
            "depth_mean": f(top, "depth"),
            "cluster_mean_depth_mean": f(top, "cluster_mean_depth"),
            "degree_gini_mean": f(top, "cluster_degree_gini"),
        }
        add_stat(top_bucket_stats, bucket(top), hit, top_values)
        add_stat(top_parent, parent_kind(top), hit, top_values)
        add_stat(top_regime, regime(top), hit, top_values)

        if gt is not None:
            cid = int(f(gt, "cid", -1))
            cs = gt_cluster_stats[cid]
            cs["n"] += 1
            cs["hit"] += int(hit)
            cs["gt_adc_rank"] += f(gt, "rank")
            cs["adc_margin"] += f(gt, "adc_distance") - f(top, "adc_distance")
            for m in GRAPH_METRICS:
                cs[m] += f(gt, m)

            adc_margin = f(gt, "adc_distance") - f(top, "adc_distance")
            exact_gap = f(top, "exact_distance") - f(gt, "exact_distance")
            recon_bias_gap = f(top, "recon_minus_exact") - f(gt, "recon_minus_exact")
            vals = {
                "gt_adc_rank_mean": f(gt, "rank"),
                "adc_margin_mean": adc_margin,
                "top_minus_gt_exact_gap_mean": exact_gap,
                "top_minus_gt_recon_bias_gap_mean": recon_bias_gap,
                "depth_mean": f(gt, "depth"),
                "cluster_mean_depth_mean": f(gt, "cluster_mean_depth"),
                "degree_gini_mean": f(gt, "cluster_degree_gini"),
            }
            add_stat(gt_bucket_stats, bucket(gt), hit, vals)
            add_stat(gt_parent, parent_kind(gt), hit, vals)
            add_stat(gt_regime, regime(gt), hit, vals)
            for m in GRAPH_METRICS:
                label = quantile_label(f(gt, m), metric_edges[m])
                add_stat(metric_bins[m], label, hit, vals)
            if hit:
                hit_margins.append(adc_margin)
            else:
                miss_margins.append(adc_margin)
                miss_parent_pair[(parent_kind(top), parent_kind(gt))] += 1
                miss_regime_pair[(regime(top), regime(gt))] += 1
                miss_by_top_exact_rank[int(f(top, "exact_rank_in_shortlist"))] += 1

    gt_cluster_rows, gt_cluster_summary = finalize_cluster_stats(gt_cluster_stats)

    return {
        "n_queries": n,
        "adc_r1": hits / n if n else 0.0,
        "gt1_in_shortlist": gt_in / n if n else 0.0,
        "top_bucket": finalize_stat(top_bucket_stats),
        "gt_bucket": finalize_stat(gt_bucket_stats),
        "top_parent": finalize_stat(top_parent),
        "gt_parent": finalize_stat(gt_parent),
        "top_regime": finalize_stat(top_regime),
        "gt_regime": finalize_stat(gt_regime),
        "gt_metric_bins": {m: finalize_stat(metric_bins[m]) for m in GRAPH_METRICS},
        "miss_parent_pair": {" -> ".join(k): v for k, v in sorted(miss_parent_pair.items(), key=lambda kv: -kv[1])},
        "miss_regime_pair": {" -> ".join(k): v for k, v in sorted(miss_regime_pair.items(), key=lambda kv: -kv[1])},
        "miss_top_exact_rank_hist_top20": dict(sorted((k, v) for k, v in miss_by_top_exact_rank.items() if k <= 20)),
        "miss_adc_margin_mean": mean(miss_margins) if miss_margins else 0.0,
        "hit_adc_margin_mean": mean(hit_margins) if hit_margins else 0.0,
        "gt_cluster_stats": gt_cluster_rows,
        "gt_cluster_summary": gt_cluster_summary,
    }


def top_items(d: dict, n: int = 8) -> list[tuple[str, dict]]:
    return sorted(d.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)[:n]


def write_md(results: dict[str, dict], out: Path) -> None:
    lines: list[str] = []
    lines.append("# STLQ Graph Hit Pattern Analysis")
    lines.append("")
    lines.append("Rows are grouped per query from `candidate_features_k100.csv`. `GT1` means the raw exact nearest neighbor when it is present in the STLQ top100 shortlist.")
    lines.append("")
    lines.append("| run | queries | ADC r@1 | GT1 in top100 | miss ADC margin mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, r in results.items():
        lines.append(f"| {name} | {r['n_queries']} | {r['adc_r1']:.4f} | {r['gt1_in_shortlist']:.4f} | {r['miss_adc_margin_mean']:.3f} |")
    lines.append("")

    for name, r in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("### GT1 Parent Source")
        lines.append("")
        lines.append("| parent | n | hit_rate | gt_adc_rank | adc_margin | exact_gap(top-gt) | recon_bias_gap(top-gt) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for k, v in top_items(r["gt_parent"], 10):
            lines.append(
                f"| {k} | {v['n']} | {v['hit_rate']:.4f} | {v.get('gt_adc_rank_mean',0):.2f} | {v.get('adc_margin_mean',0):.1f} | {v.get('top_minus_gt_exact_gap_mean',0):.1f} | {v.get('top_minus_gt_recon_bias_gap_mean',0):.1f} |"
            )
        lines.append("")
        lines.append("### Selected Top1 Parent Source")
        lines.append("")
        lines.append("| parent | n | hit_rate | top exact rank | recon_minus_exact | depth |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for k, v in top_items(r["top_parent"], 10):
            lines.append(
                f"| {k} | {v['n']} | {v['hit_rate']:.4f} | {v.get('exact_rank_mean',0):.2f} | {v.get('recon_minus_exact_mean',0):.1f} | {v.get('depth_mean',0):.2f} |"
            )
        lines.append("")
        lines.append("### GT1 Graph Regime")
        lines.append("")
        lines.append("| regime | n | hit_rate | gt_adc_rank | adc_margin | exact_gap(top-gt) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for k, v in top_items(r["gt_regime"], 10):
            lines.append(
                f"| {k} | {v['n']} | {v['hit_rate']:.4f} | {v.get('gt_adc_rank_mean',0):.2f} | {v.get('adc_margin_mean',0):.1f} | {v.get('top_minus_gt_exact_gap_mean',0):.1f} |"
            )
        lines.append("")
        lines.append("### Miss Pair: selected top1 parent -> GT1 parent")
        lines.append("")
        lines.append("| pair | count |")
        lines.append("|---|---:|")
        for k, v in list(r["miss_parent_pair"].items())[:10]:
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("### GT1 Cluster Metric Quantiles")
        lines.append("")
        for m, bins in r["gt_metric_bins"].items():
            lines.append(f"#### {m}")
            lines.append("")
            lines.append("| bin | n | hit_rate | gt_adc_rank | adc_margin |")
            lines.append("|---|---:|---:|---:|---:|")
            for k, v in sorted(bins.items()):
                lines.append(f"| {k} | {v['n']} | {v['hit_rate']:.4f} | {v.get('gt_adc_rank_mean',0):.2f} | {v.get('adc_margin_mean',0):.1f} |")
            lines.append("")
        lines.append("### GT1 Cluster-Level Correlation")
        lines.append("")
        lines.append("Weighted Pearson correlation over clusters with at least 20 GT1-in-shortlist queries. Positive `hit_rate` correlation means larger metric tends to match higher r@1 when GT1 belongs to that cluster.")
        lines.append("")
        lines.append("| metric | corr(hit_rate) | corr(gt_adc_rank) | corr(adc_margin) |")
        lines.append("|---|---:|---:|---:|")
        corr = r["gt_cluster_summary"]["correlations"]
        for m in GRAPH_METRICS:
            lines.append(
                f"| {m} | {corr['hit_rate'][m]:.4f} | {corr['gt_adc_rank_mean'][m]:.4f} | {corr['adc_margin_mean'][m]:.4f} |"
            )
        lines.append("")
        lines.append("### Best/Worst GT1 Clusters")
        lines.append("")
        lines.append("| group | cid | n | hit_rate | gt_adc_rank | mean_depth | degree_gini | linked_ratio | n_virtual |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for group, rows in [
            ("best", r["gt_cluster_summary"]["top_good_clusters_n_ge_20"]),
            ("worst", r["gt_cluster_summary"]["top_bad_clusters_n_ge_20"]),
        ]:
            for row in rows[:5]:
                lines.append(
                    f"| {group} | {int(row['cid'])} | {int(row['n'])} | {row['hit_rate']:.4f} | {row['gt_adc_rank_mean']:.2f} | {row['cluster_mean_depth']:.3f} | {row['cluster_degree_gini']:.3f} | {row['cluster_linked_ratio']:.4f} | {row['cluster_n_virtual']:.0f} |"
                )
        lines.append("")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    csvs = [Path(x) for x in args.csv]
    names = args.name or []
    if names and len(names) != len(csvs):
        raise SystemExit("--name count must match --csv count")
    if not names:
        names = [p.parents[1].name if p.parent.name == "analysis" else p.stem for p in csvs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in zip(names, csvs):
        print(f"[graph-hit] loading {name}: {path}", flush=True)
        records = load_query_records(path)
        results[name] = analyze_records(records)
    (out_dir / "graph_hit_patterns.json").write_text(json.dumps(results, indent=2))
    write_md(results, out_dir / "graph_hit_patterns.md")
    print(f"[graph-hit] wrote {out_dir / 'graph_hit_patterns.md'}")


if __name__ == "__main__":
    main()
