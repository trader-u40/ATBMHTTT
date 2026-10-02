#!/usr/bin/env python3
"""P4: Oracle-in-the-loop on the P3 LSTM. Reward -2 if payload no longer executes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from actions import ACTION_META  # noqa: E402
from dl_preprocess import load_vocab, oov_rate  # noqa: E402
from xss_env import XSSDetectEnv  # noqa: E402

LAB = SRC.parent
DATA = LAB / "data"
RUNS_P3 = LAB / "runs" / "p3"
RUNS = LAB / "runs" / "p4"
SPLIT = DATA / "splits"


def _malicious_payloads(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    return df[df["Class"] == "Malicious"]["Payloads"].astype(str).tolist()


def _make_env(payloads: list[str], sequential: bool, oracle_reward: bool) -> XSSDetectEnv:
    return XSSDetectEnv(
        payloads,
        RUNS_P3 / "vocab.json",
        RUNS_P3 / "lstm.pt",
        sequential=sequential,
        oracle_reward=oracle_reward,
    )


def train_algo(algo: str, seeds: list[int], timesteps: int) -> None:
    from stable_baselines3 import DQN, PPO

    payloads = _malicious_payloads(SPLIT / "adversarial_agents" / "train.csv")
    RUNS.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        random.seed(seed)
        env = _make_env(payloads, sequential=False, oracle_reward=True)
        if algo == "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                device="cpu",
                n_steps=256,
                batch_size=64,
            )
        elif algo == "dqn":
            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                device="cpu",
                learning_starts=500,
                buffer_size=20000,
                batch_size=64,
            )
        else:
            raise ValueError(algo)
        model.learn(total_timesteps=timesteps)
        out = RUNS / f"{algo}_oracle_seed{seed}.zip"
        model.save(str(out))
        env.close()
        print(json.dumps({"algo": algo, "seed": seed, "timesteps": timesteps, "path": str(out)}), flush=True)


def eval_zip(zip_path: Path, algo: str, payloads: list[str], vocab_set: set[str]) -> dict:
    from stable_baselines3 import DQN, PPO

    from oracle import JsdomOracle

    loader = PPO.load if algo == "ppo" else DQN.load
    env = _make_env(payloads, sequential=True, oracle_reward=False)
    model = loader(str(zip_path), device="cpu")
    n_escape = 0
    exec_alive = 0
    none_n = tok_n = 0
    actions_all: Counter[int] = Counter()
    actions_success: Counter[int] = Counter()
    with JsdomOracle() as oracle:
        for payload in payloads:
            obs, _ = env.reset(options={"payload": payload})
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            for aid in env.action_trace:
                actions_all[aid] += 1
            if env.success:
                n_escape += 1
                for aid in env.action_trace:
                    actions_success[aid] += 1
                nn, nt = oov_rate(env.escaped_payload, vocab_set)
                none_n += nn
                tok_n += nt
                if oracle.evaluate(env.escaped_payload).executed:
                    exec_alive += 1
    env.close()
    n = max(len(payloads), 1)
    hist = {
        ACTION_META[i]["name"]: actions_success.get(i, 0) for i in range(1, 28)
    }
    return {
        "n": len(payloads),
        "n_escape": n_escape,
        "er": n_escape / n,
        "exec_alive_among_escape": exec_alive,
        "tasr": exec_alive / n,
        "rr_e": (1.0 - exec_alive / n_escape) if n_escape else None,
        "or_v": (none_n / tok_n) if tok_n else None,
        "action_hist_success": hist,
        "top_actions": [
            ACTION_META[i]["name"]
            for i, _ in actions_success.most_common(8)
        ],
    }


def phase_eval(seeds: list[int], algos: list[str], max_eval: int) -> None:
    payloads = _malicious_payloads(SPLIT / "adversarial_agents" / "test.csv")
    if max_eval and max_eval < len(payloads):
        payloads = payloads[:max_eval]
    vocab = load_vocab(RUNS_P3 / "vocab.json")
    vocab_set = set(vocab) - {"None", "<pad>"}
    rows = []
    for algo in algos:
        for seed in seeds:
            zip_path = RUNS / f"{algo}_oracle_seed{seed}.zip"
            if not zip_path.exists():
                print(json.dumps({"skip": str(zip_path)}), flush=True)
                continue
            row = eval_zip(zip_path, algo, payloads, vocab_set)
            row.update({"algo": algo, "seed": seed, "oracle_in_train": True})
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("algo", "seed", "er", "tasr", "rr_e", "or_v", "top_actions")}), flush=True)
    p3_path = RUNS_P3 / "p3_eval.json"
    p3 = json.loads(p3_path.read_text(encoding="utf-8")) if p3_path.exists() else {}
    summary = {
        "p4": rows,
        "p3_ref": {
            "er_mean_20k": p3.get("er_mean_20k_three_seeds"),
            "note": p3.get("note"),
        },
        "note": "P4 R_exec: +10 only if detector evade AND JSDOM execute; -2 if ruined.",
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "p4_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])


def main() -> None:
    global RUNS, RUNS_P3
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--algo", default="ppo", help="ppo,dqn or ppo")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--max-eval", type=int, default=400)
    parser.add_argument("--p3-dir", type=Path, default=RUNS_P3)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    args = parser.parse_args()
    RUNS, RUNS_P3 = args.runs_dir, args.p3_dir
    RUNS.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    algos = [a.strip() for a in args.algo.split(",") if a.strip()]
    if args.phase in {"train", "all"}:
        for algo in algos:
            train_algo(algo, seeds, args.timesteps)
    if args.phase in {"eval", "all"}:
        phase_eval(seeds, algos, args.max_eval)


if __name__ == "__main__":
    main()
