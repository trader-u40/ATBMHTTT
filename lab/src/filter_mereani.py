#!/usr/bin/env python3
"""Filter Mereani so RR on the kept malicious set is 0 for the execution oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from oracle import JsdomOracle  # noqa: E402
from decode import decode_loop  # noqa: E402

LAB_ROOT = SRC.parent
DATA = LAB_ROOT / "data"


def load_mereani(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df["Payloads"] = df["Payloads"].astype(str)
    df["Class"] = df["Class"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "Payloads.csv")
    parser.add_argument("--output", type=Path, default=DATA / "mereani_exec_oracle.csv")
    parser.add_argument("--report", type=Path, default=DATA / "filter_report.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    df = load_mereani(args.input)
    if args.limit:
        mal = df[df["Class"] == "Malicious"].head(args.limit // 2)
        ben = df[df["Class"] == "Benign"].head(args.limit - len(mal))
        df = pd.concat([mal, ben], ignore_index=True)

    rows = []
    counts = {
        "input": int(len(df)),
        "malicious_in": int((df["Class"] == "Malicious").sum()),
        "benign_in": int((df["Class"] == "Benign").sum()),
        "malicious_exec": 0,
        "malicious_parser_only": 0,
        "malicious_neither": 0,
        "benign_exec_false_positive": 0,
        "benign_kept": 0,
        "malicious_kept": 0,
    }

    with JsdomOracle() as oracle:
        for i, rec in enumerate(df.itertuples(index=False), start=1):
            payload = rec.Payloads
            label = rec.Class
            result = oracle.evaluate(payload)
            keep = False
            if label == "Malicious":
                if result.executed:
                    counts["malicious_exec"] += 1
                    keep = True
                elif result.parser_changed:
                    counts["malicious_parser_only"] += 1
                else:
                    counts["malicious_neither"] += 1
            else:
                if result.executed:
                    counts["benign_exec_false_positive"] += 1
                else:
                    keep = True
                    counts["benign_kept"] += 1
            if keep:
                if label == "Malicious":
                    counts["malicious_kept"] += 1
                rows.append(
                    {
                        "Payloads": payload,
                        "Class": label,
                        "decoded": decode_loop(payload),
                        "executed": bool(result.executed),
                        "parser_changed": bool(result.parser_changed),
                    }
                )
            if i % 500 == 0:
                print(f"processed {i}/{len(df)} kept {len(rows)}", flush=True)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # csv cannot write NUL; HTML parsers map it to U+FFFD anyway.
    for col in ("Payloads", "decoded"):
        if col in out:
            out[col] = out[col].str.replace("\x00", "\ufffd", regex=False)
    out.to_csv(args.output, index=False)

    kept_mal = out[out["Class"] == "Malicious"] if len(out) else out
    rr = 0.0
    if len(kept_mal):
        rr = 1.0 - float(kept_mal["executed"].mean())
    counts["rr_malicious_kept"] = rr
    counts["output_rows"] = int(len(out))
    counts["w1_parser_only_ratio"] = (
        counts["malicious_parser_only"] / counts["malicious_in"]
        if counts["malicious_in"]
        else 0.0
    )
    args.report.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    if rr != 0.0:
        raise SystemExit(f"RR on kept malicious is {rr}, expected 0")


if __name__ == "__main__":
    main()
