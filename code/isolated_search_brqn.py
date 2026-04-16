import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.8/bin/python3"
DEFAULT_MODELS = ["MAB", "XgBoost", "DQN", "BRQN"]
SCRIPT_FILES = [
    "search_brqn_setup.py",
    "generate_synthetic_features.py",
    "MAB.py",
    "XgBoost.py",
    "DQN.py",
    "DRQN.py",
    "BRQN.py",
]
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


def prepare_isolated_run(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_FILES:
        shutil.copy2(BASE_DIR / name, run_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--train-users", type=int, default=140)
    parser.add_argument("--test-users", type=int, default=90)
    parser.add_argument("--min-steps", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--brqn-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--xgb-estimators", type=int, default=150)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--outroot", required=True)
    args = parser.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        run_dir = outroot / f"seed_{seed}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        prepare_isolated_run(run_dir)

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
            str(seed),
        ]
        rc, text = run(gen_cmd, gen_env, run_dir)
        (run_dir / "generate.log").write_text(text)
        if rc != 0:
            rows.append({"seed": seed, "mode": args.mode, "horizon": "nan", "model": "generate", "returncode": rc})
            continue

        for horizon in args.horizons:
            for model in args.models:
                env = os.environ.copy()
                env["EPISODE_HORIZON"] = str(horizon)
                env["SEED"] = str(seed)
                env["NUM_EPISODES"] = str(args.num_episodes)
                env["BATCH_SIZE"] = str(args.batch_size)
                env["XGB_N_ESTIMATORS"] = str(args.xgb_estimators)
                if model == "BRQN" and args.brqn_episodes is not None:
                    env["NUM_EPISODES"] = str(args.brqn_episodes)
                if model == "DRQN":
                    env["NUM_EPISODES"] = str(max(args.num_episodes - 8, 20))
                cmd = [PYTHON, f"{model}.py"]
                rc, text = run(cmd, env, run_dir)
                (run_dir / f"{model}_H{horizon}.log").write_text(text)
                metrics = parse_metrics(text)
                row = {
                    "seed": seed,
                    "mode": args.mode,
                    "horizon": horizon,
                    "model": model,
                    "returncode": rc,
                }
                row.update(metrics)
                rows.append(row)
                print(
                    f"seed={seed} H={horizon} {model} rc={rc} accepted={metrics['mean_accepted_reward']} count={metrics['accepted_episodes']}",
                    flush=True,
                )

    csv_path = outroot / "results.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["seed", "mode", "horizon", "model", "returncode"] + list(PATTERNS)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
