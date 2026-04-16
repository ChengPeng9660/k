import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.8/bin/python3"
MODELS = ["MAB", "XgBoost", "DQN", "DRQN", "BRQN"]
SCRIPTS = {
    "MAB": "MAB.py",
    "XgBoost": "XgBoost.py",
    "DQN": "DQN.py",
    "DRQN": "DRQN.py",
    "BRQN": "BRQN.py",
}
PATTERNS = {
    "mean_all_reward": r"mean for all reward:\s*([0-9.]+|nan)",
    "mean_accepted_reward": r"mean for accepted reward:\s*([0-9.]+|nan)",
    "accepted_episodes": r"accepted episodes:\s*([0-9.]+|nan)",
    "total_candidate_episodes": r"total candidate episodes:\s*([0-9.]+|nan)",
    "acceptance_rate": r"acceptance rate:\s*([0-9.]+|nan)",
}


def parse_metrics(text):
    out = {}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, text)
        out[key] = found[-1] if found else "nan"
    return out


def run(cmd, env, cwd):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--train-users", type=int, default=140)
    parser.add_argument("--test-users", type=int, default=90)
    parser.add_argument("--min-steps", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--xgb-estimators", type=int, default=150)
    parser.add_argument("--brqn-episodes", type=int, default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gen_env = os.environ.copy()
    gen_cmd = [
        PYTHON,
        "generate_synthetic_features.py",
        "--mode",
        args.mode,
        "--train-users",
        str(args.train_users),
        "--test-users",
        str(args.test_users),
        "--min-steps",
        str(args.min_steps),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
    ]
    rc, text = run(gen_cmd, gen_env, BASE_DIR)
    (outdir / "generate.log").write_text(text)
    if rc != 0:
        raise SystemExit(f"generation failed with code {rc}")

    rows = []
    for horizon in args.horizons:
        for model in args.models:
            env = os.environ.copy()
            env["EPISODE_HORIZON"] = str(horizon)
            env["SEED"] = str(args.seed)
            env["NUM_EPISODES"] = str(args.num_episodes)
            env["BATCH_SIZE"] = str(args.batch_size)
            env["XGB_N_ESTIMATORS"] = str(args.xgb_estimators)
            if model == "BRQN":
                brqn_episodes = args.brqn_episodes if args.brqn_episodes is not None else max(args.num_episodes, 40)
                env["NUM_EPISODES"] = str(brqn_episodes)
            if model == "DRQN":
                env["NUM_EPISODES"] = str(max(args.num_episodes - 8, 20))
            cmd = [PYTHON, SCRIPTS[model]]
            rc, text = run(cmd, env, BASE_DIR)
            (outdir / f"{model}_H{horizon}.log").write_text(text)
            metrics = parse_metrics(text)
            row = {"mode": args.mode, "horizon": horizon, "model": model, "returncode": rc}
            row.update(metrics)
            rows.append(row)
            print(mode_summary(row))

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["mode", "horizon", "model", "returncode"] + list(PATTERNS)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


def mode_summary(row):
    return (
        f"{row['mode']} H={row['horizon']} {row['model']} rc={row['returncode']} "
        f"accepted={row['mean_accepted_reward']} "
        f"count={row['accepted_episodes']} rate={row['acceptance_rate']}"
    )


if __name__ == "__main__":
    main()
