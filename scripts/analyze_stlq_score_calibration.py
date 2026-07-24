#!/usr/bin/env python3
"""Offline STLQ score-calibration analysis.

The script reads export_stlq_candidate_features CSV files and evaluates whether
deployable query-time features can correct STLQ shortlist ordering bias.
Outputs belong under cmake-build-release/analysis by convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FEATURES = [
    "adc_distance",
    "recon_distance",
    "recon_norm2",
    "q_dot_recon",
    "center_norm2",
    "q_dot_center",
    "q_center_distance",
    "depth",
    "parent_depth",
    "child_degree",
    "subtree_real",
    "cluster_n_real",
    "cluster_n_virtual",
    "cluster_linked_ratio",
    "cluster_mean_depth",
    "cluster_max_depth",
    "cluster_max_child_degree",
    "cluster_max_subtree_real",
    "cluster_degree_gini",
]

PARENT_KINDS = ["root", "super_root", "virtual", "real_prev_depth", "unknown"]
REGIMES = ["with_virtual", "no_virtual_deep", "no_virtual_mid", "no_virtual_shallow"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True, help="candidate_features_k100.csv")
    p.add_argument("--name", action="append", help="run label; defaults to csv parent names")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_queries", type=int, default=5000)
    p.add_argument("--ridge", type=float, default=1e-3)
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
    n_virtual = f(row, "cluster_n_virtual")
    mean_depth = f(row, "cluster_mean_depth")
    if n_virtual > 0:
        return "with_virtual"
    if mean_depth >= 3.0:
        return "no_virtual_deep"
    if mean_depth >= 2.0:
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


def bucket_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (parent_kind(row), regime(row), depth_bin(row))


def feature_vector(row: dict[str, str]) -> list[float]:
    xs = [f(row, k) for k in NUMERIC_FEATURES]
    pk = parent_kind(row)
    rg = regime(row)
    xs.extend(1.0 if pk == k else 0.0 for k in PARENT_KINDS[:-1])
    xs.extend(1.0 if rg == r else 0.0 for r in REGIMES[:-1])
    return xs


def feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES)
    names.extend(f"parent_kind={k}" for k in PARENT_KINDS[:-1])
    names.extend(f"regime={r}" for r in REGIMES[:-1])
    return names


def load_rows(csv_path: Path) -> dict[str, np.ndarray | list[tuple[str, str, str]]]:
    qids: list[int] = []
    is_gt1: list[int] = []
    adc: list[float] = []
    recon: list[float] = []
    exact: list[float] = []
    gt_in_shortlist: list[int] = []
    exact_rank: list[int] = []
    buckets: list[tuple[str, str, str]] = []
    feats: list[list[float]] = []

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            qids.append(int(row["query_id"]))
            is_gt1.append(1 if int(row.get("is_gt1", "0")) != 0 else 0)
            adc.append(f(row, "adc_distance"))
            recon.append(f(row, "recon_distance"))
            exact.append(f(row, "exact_distance"))
            gt_in_shortlist.append(1 if int(row.get("gt1_in_shortlist", "0")) != 0 else 0)
            exact_rank.append(int(float(row.get("exact_rank_in_shortlist", "0") or 0)))
            buckets.append(bucket_key(row))
            feats.append(feature_vector(row))

    return {
        "qid": np.asarray(qids, dtype=np.int32),
        "is_gt1": np.asarray(is_gt1, dtype=np.int8),
        "adc": np.asarray(adc, dtype=np.float64),
        "recon": np.asarray(recon, dtype=np.float64),
        "exact": np.asarray(exact, dtype=np.float64),
        "gt_in_shortlist": np.asarray(gt_in_shortlist, dtype=np.int8),
        "exact_rank": np.asarray(exact_rank, dtype=np.int32),
        "bucket": buckets,
        "X": np.asarray(feats, dtype=np.float64),
    }


def query_slices(qid: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    if len(qid) == 0:
        return out
    start = 0
    last = int(qid[0])
    for i in range(1, len(qid)):
        cur = int(qid[i])
        if cur != last:
            out.append((last, start, i))
            start = i
            last = cur
    out.append((last, start, len(qid)))
    return out


def recall_for_score(qid: np.ndarray, is_gt1: np.ndarray, score: np.ndarray, split: str, train_queries: int) -> dict[str, float]:
    total = 0
    hit = 0
    gt_in = 0
    for q, lo, hi in query_slices(qid):
        in_train = q < train_queries
        if split == "train" and not in_train:
            continue
        if split == "test" and in_train:
            continue
        total += 1
        best = lo + int(np.argmin(score[lo:hi]))
        hit += int(is_gt1[best] != 0)
        gt_in += int(np.any(is_gt1[lo:hi] != 0))
    return {
        "queries": total,
        "r1": hit / total if total else 0.0,
        "gt1_in_shortlist": gt_in / total if total else 0.0,
    }


def build_bucket_correction(data: dict[str, np.ndarray | list[tuple[str, str, str]]], train_queries: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    qid = data["qid"]
    adc = data["adc"]
    exact = data["exact"]
    buckets = data["bucket"]
    assert isinstance(qid, np.ndarray) and isinstance(adc, np.ndarray) and isinstance(exact, np.ndarray)
    assert isinstance(buckets, list)

    sums: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    global_sum = 0.0
    global_n = 0
    for i, q in enumerate(qid):
        if int(q) >= train_queries:
            continue
        delta = float(exact[i] - adc[i])
        sums[buckets[i]][0] += delta
        sums[buckets[i]][1] += 1.0
        global_sum += delta
        global_n += 1

    global_mean = global_sum / max(global_n, 1)
    means = {k: (v[0] / v[1]) for k, v in sums.items() if v[1] >= 50}
    correction = np.empty_like(adc)
    for i, b in enumerate(buckets):
        correction[i] = means.get(b, global_mean)

    stats = {
        "|".join(k): {"mean_delta": means.get(k, global_mean), "n_train": sums[k][1]}
        for k in sorted(sums)
    }
    return adc + correction, stats


def fit_ridge_score(data: dict[str, np.ndarray | list[tuple[str, str, str]]], train_queries: int, ridge: float) -> tuple[np.ndarray, dict[str, float | list[tuple[str, float]]]]:
    qid = data["qid"]
    X = data["X"]
    y = data["exact"]
    assert isinstance(qid, np.ndarray) and isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    train_mask = qid < train_queries
    X_train = X[train_mask]
    y_train = y[train_mask]

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Xs = (X_train - mu) / sigma
    A = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    reg = np.eye(A.shape[1]) * ridge
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y_train)

    X_all = (X - mu) / sigma
    score = np.concatenate([np.ones((X_all.shape[0], 1)), X_all], axis=1) @ coef
    names = ["intercept"] + feature_names()
    top_coef = sorted(zip(names, coef.tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    return score, {"top_abs_coefficients": top_coef}


def summarize_one(name: str, csv_path: Path, train_queries: int, ridge: float) -> dict:
    data = load_rows(csv_path)
    qid = data["qid"]
    is_gt1 = data["is_gt1"]
    adc = data["adc"]
    recon = data["recon"]
    exact = data["exact"]
    assert isinstance(qid, np.ndarray)
    assert isinstance(is_gt1, np.ndarray)
    assert isinstance(adc, np.ndarray)
    assert isinstance(recon, np.ndarray)
    assert isinstance(exact, np.ndarray)

    bucket_score, bucket_stats = build_bucket_correction(data, train_queries)
    ridge_score, ridge_stats = fit_ridge_score(data, train_queries, ridge)

    scores = {
        "adc_current": adc,
        "recon_distance": recon,
        "bucket_calibrated_adc": bucket_score,
        "ridge_deployable_features": ridge_score,
        "oracle_exact": exact,
    }

    recalls = {}
    for score_name, score in scores.items():
        recalls[score_name] = {
            "all": recall_for_score(qid, is_gt1, score, "all", train_queries),
            "train": recall_for_score(qid, is_gt1, score, "train", train_queries),
            "test": recall_for_score(qid, is_gt1, score, "test", train_queries),
        }

    return {
        "name": name,
        "csv": str(csv_path),
        "n_rows": int(len(qid)),
        "n_queries": int(len(query_slices(qid))),
        "train_queries": train_queries,
        "recall": recalls,
        "bucket_count": len(bucket_stats),
        "bucket_stats": bucket_stats,
        "ridge": ridge_stats,
    }


def write_markdown(results: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# STLQ Score Calibration Analysis")
    lines.append("")
    lines.append("This is an offline deployability check. Labels use raw exact distance, but score inputs are restricted to ADC/reconstruction/IVF-center/graph fields available at query time.")
    lines.append("")
    lines.append("| run | split | ADC r@1 | recon-dist r@1 | bucket-calibrated r@1 | ridge r@1 | oracle exact r@1 | gt1 in top100 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        rec = r["recall"]
        for split in ["all", "train", "test"]:
            lines.append(
                "| {name} | {split} | {adc:.4f} | {recon:.4f} | {bucket:.4f} | {ridge:.4f} | {oracle:.4f} | {gt:.4f} |".format(
                    name=r["name"],
                    split=split,
                    adc=rec["adc_current"][split]["r1"],
                    recon=rec["recon_distance"][split]["r1"],
                    bucket=rec["bucket_calibrated_adc"][split]["r1"],
                    ridge=rec["ridge_deployable_features"][split]["r1"],
                    oracle=rec["oracle_exact"][split]["r1"],
                    gt=rec["adc_current"][split]["gt1_in_shortlist"],
                )
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `ADC r@1` is the current STLQ shortlist ordering reproduced from the CSV.")
    lines.append("- `recon-dist r@1` reranks by exact distance to the decoded reconstructed vector, not the raw vector.")
    lines.append("- `bucket-calibrated` adds train-split mean `exact_distance - adc_distance` by `(parent_kind, graph_regime, depth_bin)`.")
    lines.append("- `ridge` is a simple linear regression to raw exact distance using only deployable features; query ids below `train_queries` are train, the rest are test.")
    lines.append("- `oracle exact` reranks by raw exact distance and is an upper bound, not deployable.")
    lines.append("")
    lines.append("## Ridge Coefficients")
    lines.append("")
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append("| feature | coefficient |")
        lines.append("|---|---:|")
        for k, v in r["ridge"]["top_abs_coefficients"]:
            lines.append(f"| `{k}` | {v:.6g} |")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


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

    results = []
    for name, csv_path in zip(names, csvs):
        print(f"[analyze] loading {name}: {csv_path}", flush=True)
        results.append(summarize_one(name, csv_path, args.train_queries, args.ridge))

    (out_dir / "score_calibration_analysis.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, out_dir / "score_calibration_analysis.md")
    print(f"[analyze] wrote {out_dir / 'score_calibration_analysis.md'}")


if __name__ == "__main__":
    main()
