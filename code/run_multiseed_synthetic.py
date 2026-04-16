import argparse
import csv
import os
import re
import subprocess
import sys
import copy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BASE_DIR / "reproduction_outputs" / "multiseed_h1"
PYTHON = sys.executable

MODEL_CONFIGS = {
    "MAB": {"cmd": [PYTHON, "MAB.py"], "env": {"EPISODE_HORIZON": "1"}},
    "XgBoost": {
        "cmd": [PYTHON, "XgBoost.py"],
        "env": {"EPISODE_HORIZON": "1", "XGB_N_ESTIMATORS": "150"},
    },
    "DQN": {
        "cmd": [PYTHON, "DQN.py"],
        "env": {"EPISODE_HORIZON": "1", "NUM_EPISODES": "20", "BATCH_SIZE": "128"},
    },
    "DNN_BayesianUpdate": {
        "cmd": [PYTHON, "DNN_BayesianUpdate.py"],
        "env": {"EPISODE_HORIZON": "1", "NUM_EPISODES": "20", "BATCH_SIZE": "128"},
    },
    "BDQN": {
        "cmd": [PYTHON, "BDQN.py"],
        "env": {"EPISODE_HORIZON": "1", "NUM_EPISODES": "10", "BATCH_SIZE": "64"},
    },
    "DRQN": {
        "cmd": [PYTHON, "DRQN.py"],
        "env": {"EPISODE_HORIZON": "1", "NUM_EPISODES": "10", "BATCH_SIZE": "64"},
    },
    "RNN_BayesianUpdate": {
        "cmd": [PYTHON, "RNN_BayesianUpdate.py"],
        "env": {"EPISODE_HORIZON": "1", "NUM_EPISODES": "10", "BATCH_SIZE": "64"},
    },
    "BRQN": {
        "cmd": [PYTHON, "BRQN.py"],
        "env": {"EPISODE_HORIZON": "1", "NUM_EPISODES": "10", "BATCH_SIZE": "64"},
    },
}

PATTERNS = {
    "mean_all_reward": r"mean for all reward:\s*([0-9.]+|nan)",
    "mean_accepted_reward": r"mean for accepted reward:\s*([0-9.]+|nan)",
    "accepted_episodes": r"accepted episodes:\s*([0-9.]+|nan)",
    "total_candidate_episodes": r"total candidate episodes:\s*([0-9.]+|nan)",
    "acceptance_rate": r"acceptance rate:\s*([0-9.]+|nan)",
}


def parse_metrics(text):
    metrics = {}
    for key, pattern in PATTERNS.items():
        match = re.findall(pattern, text)
        metrics[key] = match[-1] if match else "nan"
    return metrics


def run_cmd(cmd, env, log_path):
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout)
    return proc.returncode, parse_metrics(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-users", type=int, default=80)
    parser.add_argument("--test-users", type=int, default=30)
    parser.add_argument("--min-steps", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--mode", choices=["default", "brqn_favor", "brqn_linear_sparse", "brqn_mechanism", "brqn_order_uncertainty", "brqn_horizon_support", "brqn_presession_linear", "brqn_regime_action", "brqn_user_mapping", "brqn_simple"], default="default")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.outdir:
        out_dir = Path(args.outdir)
    elif args.mode == "brqn_favor":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_favor"
    elif args.mode == "brqn_linear_sparse":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_linear_sparse"
    elif args.mode == "brqn_mechanism":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_mechanism"
    elif args.mode == "brqn_order_uncertainty":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_order_uncertainty"
    elif args.mode == "brqn_horizon_support":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_horizon_support"
    elif args.mode == "brqn_presession_linear":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_presession_linear"
    elif args.mode == "brqn_regime_action":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_regime_action"
    elif args.mode == "brqn_user_mapping":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_user_mapping"
    elif args.mode == "brqn_simple":
        out_dir = BASE_DIR / "reproduction_outputs" / "multiseed_h1_brqn_simple"
    else:
        out_dir = DEFAULT_OUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model_configs = copy.deepcopy(MODEL_CONFIGS)

    for cfg in model_configs.values():
        cfg["env"]["EPISODE_HORIZON"] = str(args.horizon)

    if args.mode == "brqn_favor":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "30"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "15"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "15"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "15"
    elif args.mode == "brqn_mechanism":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "24"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "18"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "18"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "18"
    elif args.mode == "brqn_order_uncertainty":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "28"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "18"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "18"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "18"
    elif args.mode == "brqn_horizon_support":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "32"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "20"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "20"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "20"
    elif args.mode == "brqn_presession_linear":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "36"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "20"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "20"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "20"
    elif args.mode == "brqn_regime_action":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "40"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "22"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "22"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "22"
    elif args.mode == "brqn_user_mapping":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "44"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "24"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "24"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "24"
    elif args.mode == "brqn_simple":
        model_configs["BRQN"]["env"]["NUM_EPISODES"] = "44"
        model_configs["DRQN"]["env"]["NUM_EPISODES"] = "24"
        model_configs["RNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "24"
        model_configs["DNN_BayesianUpdate"]["env"]["NUM_EPISODES"] = "24"

    for seed in args.seeds:
        gen_env = os.environ.copy()
        gen_cmd = [
            sys.executable,
            "generate_synthetic_features.py",
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
            "--mode",
            args.mode,
        ]
        subprocess.run(gen_cmd, cwd=str(BASE_DIR), env=gen_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for model, cfg in model_configs.items():
            env = os.environ.copy()
            env.update(cfg["env"])
            env["SEED"] = str(seed)
            log_path = out_dir / f"{model}_seed{seed}.log"
            returncode, metrics = run_cmd(cfg["cmd"], env, log_path)
            row = {"model": model, "seed": seed, "returncode": returncode}
            row.update(metrics)
            rows.append(row)
            print(model, seed, returncode, metrics["mean_accepted_reward"])

    csv_path = out_dir / "multiseed_results.csv"
    fieldnames = ["model", "seed", "returncode"] + list(PATTERNS)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
