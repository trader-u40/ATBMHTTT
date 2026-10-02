#!/usr/bin/env python3
"""Paired P3 vs P4 ER per seed, read from p3_eval.json / p4_eval.json. No scipy required."""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def signed_rank(diffs: list[float]) -> dict:
    pairs = sorted((abs(d), d) for d in diffs if d != 0)
    ranks = []
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks.append(avg if pairs[k][1] > 0 else -avg)
        i = j + 1
    w_plus = sum(r for r in ranks if r > 0)
    w_minus = -sum(r for r in ranks if r < 0)
    n = len(pairs)
    abs_ranks = [abs(r) for r in ranks]
    obs = max(w_plus, w_minus)
    extreme = sum(
        1
        for signs in product([-1, 1], repeat=n)
        if sum(r for s, r in zip(signs, abs_ranks) if s > 0) >= obs - 1e-12
    )
    p = min(1.0, 2 * extreme / (2**n)) if n else 1.0
    # Smallest achievable two-sided p with n nonzero pairs is 2/2^n.
    return {
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_two_sided_exact": p,
        "min_achievable_p": (2 / 2**n) if n else 1.0,
    }


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", type=Path, default=LAB / "runs" / "p3" / "p3_eval.json")
    ap.add_argument("--p4", type=Path, default=LAB / "runs" / "p4" / "p4_eval.json")
    ap.add_argument("--timesteps", type=int, default=20000, help="P3 rows to use")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--out", type=Path, default=LAB / "data" / "wilcoxon_p3p4.json")
    args = ap.parse_args()

    p3 = json.loads(args.p3.read_text(encoding="utf-8"))
    p4 = json.loads(args.p4.read_text(encoding="utf-8"))
    p3_rows = {
        r["seed"]: r for r in p3.get("ppo", p3.get("seeds", [])) if r.get("timesteps", args.timesteps) == args.timesteps
    }
    p4_rows = {r["seed"]: r for r in p4["p4"] if r.get("algo", "ppo") == args.algo}
    seeds = sorted(set(p3_rows) & set(p4_rows))
    if not seeds:
        raise SystemExit("no common seeds between P3 and P4")

    per_seed = []
    for s in seeds:
        a, b = p3_rows[s], p4_rows[s]
        per_seed.append(
            {
                "seed": s,
                "p3_escape": a["n_escape"],
                "p3_er": a["er"],
                "p3_er_ci95": wilson(a["n_escape"], a["n"]),
                "p4_escape": b["n_escape"],
                "p4_er": b["er"],
                "p4_er_ci95": wilson(b["n_escape"], b["n"]),
            }
        )
    diffs = [r["p3_er"] - r["p4_er"] for r in per_seed]
    baseline = p3.get("baseline_fn_unmutated", {})
    out = {
        "metric": "ER on agent-test payloads",
        "p3_source": str(args.p3.relative_to(LAB) if args.p3.is_relative_to(LAB) else args.p3),
        "p4_source": str(args.p4.relative_to(LAB) if args.p4.is_relative_to(LAB) else args.p4),
        "p3_timesteps": args.timesteps,
        "algo": args.algo,
        "detector_fn_floor": baseline,
        "per_seed": per_seed,
        "diffs_p3_minus_p4": diffs,
        "wilcoxon": signed_rank(diffs),
        "reading": (
            "Descriptive only. With few seeds the exact test cannot reach 0.05 "
            "(see min_achievable_p); compare ER against detector_fn_floor and the per-seed Wilson CIs."
        ),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
