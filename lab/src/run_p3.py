#!/usr/bin/env python3
"""P3 protocol replication: LSTM detector + PPO (no oracle in reward)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from dl_preprocess import (  # noqa: E402
    build_vocab,
    encode_payload,
    load_vocab,
    oov_rate,
    save_vocab,
)
from lstm_detector import LSTMDetector  # noqa: E402
from xss_env import XSSDetectEnv  # noqa: E402

LAB = SRC.parent
DATA = LAB / "data"
RUNS = LAB / "runs" / "p3"
SPLIT = DATA / "splits"


def _mal_ben(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    payloads = df["Payloads"].astype(str).tolist()
    labels = [1 if c == "Malicious" else 0 for c in df["Class"].tolist()]
    return payloads, labels


def phase_vocab() -> None:
    train = pd.read_csv(SPLIT / "detectors" / "train.csv")
    vocab = build_vocab(train["Payloads"].astype(str).tolist(), keep_ratio=0.1)
    RUNS.mkdir(parents=True, exist_ok=True)
    save_vocab(RUNS / "vocab.json", vocab)
    print(
        json.dumps(
            {
                "vocab_size": len(vocab),
                "none_in_vocab": "None" in vocab,
                "pad": vocab[0],
            }
        )
    )


def _tensorize(csv_path: Path, vocab: list[str]) -> TensorDataset:
    df = pd.read_csv(csv_path)
    payloads, labels = _mal_ben(df)
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    none_id = token_to_id["None"]
    xs = torch.stack([encode_payload(p, token_to_id, none_id) for p in payloads])
    ys = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(xs, ys)


def phase_detector(epochs: int, patience: int, batch_size: int, lr: float, seed: int) -> None:
    torch.manual_seed(seed)
    vocab = load_vocab(RUNS / "vocab.json")
    train_ds = _tensorize(SPLIT / "detectors" / "train.csv", vocab)
    val_ds = _tensorize(SPLIT / "detectors" / "val.csv", vocab)
    test_ds = _tensorize(SPLIT / "detectors" / "test.csv", vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    model = LSTMDetector(len(vocab), embedding_dim=8)
    pad_id = {tok: i for i, tok in enumerate(vocab)}["<pad>"]
    # Paper uses SGD 1e-3; on this split it stuck at 50%. Adam is a documented deviation.
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb, pad_id=pad_id), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
        train_loss = total / len(train_ds)
        val_metrics = _eval_loader(model, val_loader, pad_id=pad_id)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f}",
            flush=True,
        )
        if val_metrics["loss"] + 1e-6 < best_val:
            best_val = val_metrics["loss"]
            stale = 0
            torch.save(model.state_dict(), RUNS / "lstm.pt")
        else:
            stale += 1
            if stale > patience:
                print("early_stop", epoch)
                break
    model.load_state_dict(torch.load(RUNS / "lstm.pt"))
    test_metrics = _eval_loader(model, DataLoader(test_ds, batch_size=64), pad_id=pad_id)
    report = {
        "vocab_size": len(vocab),
        "embedding_dim": 8,
        "max_length": 40,
        "optimizer": "Adam",
        "optimizer_note": "Paper SGD 1e-3 stalled at 50% acc; Adam is a documented deviation.",
        "best_val_loss": best_val,
        "test": test_metrics,
        "history_tail": history[-5:],
    }
    (RUNS / "detector_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


@torch.no_grad()
def _eval_loader(model: LSTMDetector, loader: DataLoader, pad_id: int = 0) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    loss_sum = 0.0
    n = 0
    crit = nn.BCELoss(reduction="sum")
    for xb, yb in loader:
        prob = model(xb, pad_id=pad_id)
        loss_sum += float(crit(prob, yb).item())
        pred = (prob >= 0.5).to(torch.int)
        gold = yb.to(torch.int)
        tp += int(((pred == 1) & (gold == 1)).sum())
        fp += int(((pred == 1) & (gold == 0)).sum())
        tn += int(((pred == 0) & (gold == 0)).sum())
        fn += int(((pred == 0) & (gold == 1)).sum())
        n += len(xb)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "loss": loss_sum / max(n, 1),
        "acc": (tp + tn) / max(n, 1),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def phase_ppo(seeds: list[int], timesteps: int) -> None:
    from stable_baselines3 import PPO

    train_df = pd.read_csv(SPLIT / "adversarial_agents" / "train.csv")
    payloads = train_df[train_df["Class"] == "Malicious"]["Payloads"].astype(str).tolist()
    vocab_path = RUNS / "vocab.json"
    ckpt = RUNS / "lstm.pt"
    for seed in seeds:
        random.seed(seed)
        env = XSSDetectEnv(payloads, vocab_path, ckpt)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device="cpu",
            n_steps=256,
            batch_size=64,
        )
        model.learn(total_timesteps=timesteps)
        out = RUNS / f"ppo_seed{seed}.zip"
        model.save(str(out))
        print(json.dumps({"seed": seed, "timesteps": timesteps, "path": str(out)}), flush=True)


def phase_eval(seeds: list[int], max_eval: int) -> None:
    from stable_baselines3 import PPO

    from oracle import JsdomOracle  # local execute oracle for RR(E)

    test_df = pd.read_csv(SPLIT / "adversarial_agents" / "test.csv")
    payloads = test_df[test_df["Class"] == "Malicious"]["Payloads"].astype(str).tolist()
    if max_eval and max_eval < len(payloads):
        payloads = payloads[:max_eval]
    vocab = load_vocab(RUNS / "vocab.json")
    vocab_set = set(vocab) - {"None", "<pad>"}
    rows = []
    escaped_all = []
    with JsdomOracle() as oracle:
        for seed in seeds:
            zip_path = RUNS / f"ppo_seed{seed}.zip"
            env = XSSDetectEnv(
                payloads,
                RUNS / "vocab.json",
                RUNS / "lstm.pt",
                sequential=True,
            )
            model = PPO.load(str(zip_path), device="cpu")
            n_escape = 0
            escaped = []
            none_n = tok_n = 0
            exec_alive = 0
            for payload in payloads:
                obs, _ = env.reset(options={"payload": payload})
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated
                if env.success:
                    n_escape += 1
                    escaped.append(env.escaped_payload)
                    nn, nt = oov_rate(env.escaped_payload, vocab_set)
                    none_n += nn
                    tok_n += nt
                    if oracle.evaluate(env.escaped_payload).executed:
                        exec_alive += 1
            er = n_escape / max(len(payloads), 1)
            rr = 1.0 - (exec_alive / n_escape) if n_escape else None
            orate = none_n / tok_n if tok_n else None
            row = {
                "seed": seed,
                "n": len(payloads),
                "n_escape": n_escape,
                "er": er,
                "rr_e": rr,
                "or_v": orate,
                "exec_alive_among_escape": exec_alive,
            }
            rows.append(row)
            escaped_all.extend(escaped)
            print(json.dumps(row), flush=True)
    summary = {
        "seeds": rows,
        "er_mean": sum(r["er"] for r in rows) / len(rows),
        "note": "P3: no oracle in PPO reward. High ER + high OR(V) = TH2 phenomenon.",
    }
    (RUNS / "p3_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if escaped_all:
        pd.DataFrame({"Payloads": escaped_all}).to_csv(RUNS / "escaped.csv", index=False)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["vocab", "detector", "ppo", "eval", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-seeds", default="42,43,44")
    parser.add_argument("--timesteps", type=int, default=25000)
    parser.add_argument("--max-eval", type=int, default=400, help="cap agent test size")
    args = parser.parse_args()
    seeds = [int(x) for x in args.ppo_seeds.split(",") if x.strip()]
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.phase in {"vocab", "all"}:
        phase_vocab()
    if args.phase in {"detector", "all"}:
        phase_detector(args.epochs, args.patience, args.batch_size, args.lr, args.seed)
    if args.phase in {"ppo", "all"}:
        phase_ppo(seeds, args.timesteps)
    if args.phase in {"eval", "all"}:
        phase_eval(seeds, args.max_eval)


if __name__ == "__main__":
    main()
