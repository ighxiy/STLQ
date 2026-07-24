#!/usr/bin/env python3
"""Recompute SIFT1M groundtruth with exact L2 (FAISS IndexFlatL2) and compare.

This is intended to validate whether an existing *.ivecs groundtruth matches
FAISS exact brute-force kNN for the provided base/query fvecs.

Default paths match common pq-dataset/sift layout.

Notes
- ivecs/fvecs format: each row is [int32 d][d values].
- Groundtruth ivecs stores neighbor ids, so d == k (typically 100).
- FAISS indices are 0-based.

Examples
  # Quick sanity: compare first 20 queries
  python3 tools/verify_sift_gt.py --query-limit 20 --threads 32

  # Full compare (10K queries) and write a new GT file for archival
  python3 tools/verify_sift_gt.py --threads 32 --write-out
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit(
        "Failed to import faiss. Install faiss-cpu (e.g. pip install faiss-cpu).\n" + str(e)
    )


def read_ivecs(path: str, limit: int | None = None) -> np.ndarray:
    """Read ivecs into int32 array of shape (n, d)."""
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise RuntimeError(f"Empty or invalid ivecs: {path}")
        d = struct.unpack("i", header)[0]
    sz = os.path.getsize(path)
    row_bytes = 4 + 4 * d
    if sz % row_bytes != 0:
        raise RuntimeError(f"Invalid ivecs size: {path} size={sz} row_bytes={row_bytes}")
    n = sz // row_bytes
    if limit is not None:
        n = min(int(limit), int(n))

    out = np.empty((n, d), dtype=np.int32)
    with open(path, "rb") as f:
        for i in range(n):
            dd = struct.unpack("i", f.read(4))[0]
            if dd != d:
                raise RuntimeError(f"Inconsistent d at row {i}: {dd} != {d}")
            out[i, :] = np.fromfile(f, dtype=np.int32, count=d)
    return out


def read_fvecs(path: str, limit: int | None = None) -> np.ndarray:
    """Read fvecs into float32 array of shape (n, d)."""
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise RuntimeError(f"Empty or invalid fvecs: {path}")
        d = struct.unpack("i", header)[0]
    sz = os.path.getsize(path)
    row_bytes = 4 + 4 * d
    if sz % row_bytes != 0:
        raise RuntimeError(f"Invalid fvecs size: {path} size={sz} row_bytes={row_bytes}")
    n = sz // row_bytes
    if limit is not None:
        n = min(int(limit), int(n))

    out = np.empty((n, d), dtype=np.float32)
    with open(path, "rb") as f:
        for i in range(n):
            dd = struct.unpack("i", f.read(4))[0]
            if dd != d:
                raise RuntimeError(f"Inconsistent d at row {i}: {dd} != {d}")
            out[i, :] = np.fromfile(f, dtype=np.float32, count=d)
    return out


def write_ivecs(path: str, ids: np.ndarray) -> None:
    ids = np.asarray(ids)
    if ids.ndim != 2:
        raise ValueError("ids must be 2D")
    n, k = ids.shape
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", k))
            ids[i].astype(np.int32, copy=False).tofile(f)
    print(f"wrote {path}: shape=({n}, {k})")


@dataclass
class CompareStats:
    nq: int
    k: int
    exact_order_rows: int
    exact_order_all: bool
    same_set_rows: int
    same_set_all: bool
    top1_match_rows: int
    mismatched_order_rows: int
    mismatched_set_rows: int


def compare_gt(gt_ref: np.ndarray, gt_new: np.ndarray) -> CompareStats:
    if gt_ref.shape != gt_new.shape:
        raise ValueError(f"shape mismatch: ref={gt_ref.shape}, new={gt_new.shape}")

    # 1) Exact order match (strict element-wise)
    exact_order = np.all(gt_ref == gt_new, axis=1)
    exact_order_rows = int(np.sum(exact_order))

    # 2) Same set match (ignore ordering): compare sorted ids per row
    ref_sorted = np.sort(gt_ref, axis=1)
    new_sorted = np.sort(gt_new, axis=1)
    same_set = np.all(ref_sorted == new_sorted, axis=1)
    same_set_rows = int(np.sum(same_set))

    # 3) Top-1 match
    top1_match = (gt_ref[:, 0] == gt_new[:, 0])
    top1_match_rows = int(np.sum(top1_match))

    mismatched_order_rows = int(gt_ref.shape[0] - exact_order_rows)
    mismatched_set_rows = int(gt_ref.shape[0] - same_set_rows)

    return CompareStats(
        nq=int(gt_ref.shape[0]),
        k=int(gt_ref.shape[1]),
        exact_order_rows=exact_order_rows,
        exact_order_all=(mismatched_order_rows == 0),
        same_set_rows=same_set_rows,
        same_set_all=(mismatched_set_rows == 0),
        top1_match_rows=top1_match_rows,
        mismatched_order_rows=mismatched_order_rows,
        mismatched_set_rows=mismatched_set_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default="/media/dell01/hdd0/download/pq-dataset/sift",
        help="Directory containing sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs",
    )
    p.add_argument("--base", default="sift_base.fvecs")
    p.add_argument("--query", default="sift_query.fvecs")
    p.add_argument("--gt", default="sift_groundtruth.ivecs")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--query-limit", type=int, default=0, help="If >0, only compare first N queries")
    p.add_argument("--add-batch", type=int, default=200_000)
    p.add_argument("--query-batch", type=int, default=200)
    p.add_argument("--write-out", action="store_true", help="Write recomputed GT to <gt>.faiss.ivecs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = os.path.join(args.data_dir, args.base)
    query_path = os.path.join(args.data_dir, args.query)
    gt_path = os.path.join(args.data_dir, args.gt)

    qlim = int(args.query_limit)
    query_limit = qlim if qlim > 0 else None

    print("=== SIFT GT verify (exact L2 via FAISS IndexFlatL2) ===")
    print(f"data_dir={args.data_dir}")
    print(f"base={base_path}")
    print(f"query={query_path}")
    print(f"gt_ref={gt_path}")
    if query_limit is not None:
        print(f"query_limit={query_limit}")

    t0 = time.time()
    gt_ref = read_ivecs(gt_path, limit=query_limit)
    k = int(gt_ref.shape[1])
    xq = read_fvecs(query_path, limit=query_limit)
    xb = read_fvecs(base_path)
    print(f"loaded: xb={xb.shape} xq={xq.shape} gt_ref={gt_ref.shape} in {time.time()-t0:.1f}s")

    if xq.shape[1] != xb.shape[1]:
        raise RuntimeError(f"dim mismatch: xb.d={xb.shape[1]} xq.d={xq.shape[1]}")

    d = int(xb.shape[1])

    # Thread control
    max_threads = int(faiss.omp_get_max_threads())
    n_threads = int(args.threads)
    if n_threads <= 0:
        n_threads = max_threads
    n_threads = min(n_threads, max_threads)
    faiss.omp_set_num_threads(n_threads)
    print(f"faiss omp threads: {n_threads} (max={max_threads})")

    # Build index
    print(f"building IndexFlatL2: n={xb.shape[0]} d={d}")
    t1 = time.time()
    index = faiss.IndexFlatL2(d)
    add_batch = max(1, int(args.add_batch))
    n = xb.shape[0]
    for i in range(0, n, add_batch):
        end = min(i + add_batch, n)
        index.add(xb[i:end])
        if end == n or (end // add_batch) % 5 == 0:
            print(f"  added {end}/{n} ({end*100/n:.0f}%)")
    print(f"index ready in {time.time()-t1:.1f}s, ntotal={index.ntotal}")

    # Search
    nq = xq.shape[0]
    print(f"searching: nq={nq} k={k}")
    t2 = time.time()
    qbatch = max(1, int(args.query_batch))
    I_parts: list[np.ndarray] = []
    for qi in range(0, nq, qbatch):
        qe = min(qi + qbatch, nq)
        _, Ii = index.search(xq[qi:qe], k)
        I_parts.append(Ii.astype(np.int32, copy=False))
        if qe == nq or (qe // qbatch) % 10 == 0:
            print(f"  searched {qe}/{nq} ({time.time()-t2:.1f}s)")
    gt_new = np.vstack(I_parts)
    print(f"search done in {time.time()-t2:.1f}s")

    # Basic sanity
    assert gt_new.min() >= 0
    assert gt_new.max() < xb.shape[0]

    stats = compare_gt(gt_ref, gt_new)
    print("=== Compare result ===")
    print(f"top1_match_rows={stats.top1_match_rows}/{stats.nq}")
    print(
        f"exact_order_rows={stats.exact_order_rows}/{stats.nq} "
        f"(mismatched_order_rows={stats.mismatched_order_rows})"
    )
    print(
        f"same_set_rows={stats.same_set_rows}/{stats.nq} "
        f"(mismatched_set_rows={stats.mismatched_set_rows})"
    )

    if not stats.exact_order_all:
        # Print first few order mismatches (but possibly same set)
        mism = np.nonzero(np.any(gt_ref != gt_new, axis=1))[0]
        show = mism[:5]
        print(f"first order-mismatched query rows: {show.tolist()}")
        for r in show:
            ref = gt_ref[r]
            new = gt_new[r]
            pos = int(np.nonzero(ref != new)[0][0])
            print(f"  row={int(r)} first_diff_pos={pos} ref={int(ref[pos])} new={int(new[pos])}")

    if not stats.same_set_all:
        mism = np.nonzero(np.any(np.sort(gt_ref, axis=1) != np.sort(gt_new, axis=1), axis=1))[0]
        show = mism[:5]
        print(f"first set-mismatched query rows: {show.tolist()}")

    if args.write_out:
        out_path = gt_path + ".faiss.ivecs"
        write_ivecs(out_path, gt_new)


if __name__ == "__main__":
    main()
