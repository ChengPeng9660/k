import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

MODEL_SCRIPTS = {
    "BRQN": "BRQN.py",
    "BDQN": "BDQN.py",
    "DRQN": "DRQN.py",
    "DQN": "DQN.py",
    "DNN_BayesianUpdate": "DNN_BayesianUpdate.py",
    "RNN_BayesianUpdate": "RNN_BayesianUpdate.py",
    "MAB": "MAB.py",
    "XgBoost": "XgBoost.py",
}

REFERENCE_LOGS = {
    "BRQN": "main_model_log(H={h}).txt",
    "BDQN": "BDQN_log(H={h}).txt",
    "DRQN": "DRQN_log(H={h}).txt",
    "DQN": "DQN_log(H={h}).txt",
    "DNN_BayesianUpdate": "DNN_BayesianUpdate_log(H={h}).txt",
    "RNN_BayesianUpdate": "RNN_BayesianUpdate_log(H={h}).txt",
    "MAB": "MAB_log(H={h}).txt",
    "XgBoost": "XgBoost_log(H={h}).txt",
}

FLOAT_PATTERNS = {
    "mean_all_reward": r"mean for all reward:\s*([0-9.]+)",
    "mean_accepted_reward": r"mean for accepted reward:\s*([0-9.]+)",
    "top1_mean_before": r"Top 1 Mean \(Before Update\):\s*([0-9.]+)",
    "top1_std_before": r"Top 1 STD \(Before Update\):\s*([0-9.]+)",
    "top2_mean_before": r"Top 2 Mean \(Before Update\):\s*([0-9.]+)",
    "top2_std_before": r"Top 2 STD \(Before Update\):\s*([0-9.]+)",
    "top3_mean_before": r"Top 3 Mean \(Before Update\):\s*([0-9.]+)",
    "top3_std_before": r"Top 3 STD \(Before Update\):\s*([0-9.]+)",
    "top1_mean_after_pi": r"Top 1 Mean \(After Update via PI\):\s*([0-9.]+)",
    "top1_std_after_pi": r"Top 1 STD \(After Update via PI\):\s*([0-9.]+)",
    "top2_mean_after_pi": r"Top 2 Mean \(After Update via PI\):\s*([0-9.]+)",
    "top2_std_after_pi": r"Top 2 STD \(After Update via PI\):\s*([0-9.]+)",
    "top3_mean_after_pi": r"Top 3 Mean \(After Update via PI\):\s*([0-9.]+)",
    "top3_std_after_pi": r"Top 3 STD \(After Update via PI\):\s*([0-9.]+)",
    "top1_mean_after_ei": r"Top 1 Mean \(After Update via EI\):\s*([0-9.]+)",
    "top1_std_after_ei": r"Top 1 STD \(After Update via EI\):\s*([0-9.]+)",
    "top2_mean_after_ei": r"Top 2 Mean \(After Update via EI\):\s*([0-9.]+)",
    "top2_std_after_ei": r"Top 2 STD \(After Update via EI\):\s*([0-9.]+)",
    "top3_mean_after_ei": r"Top 3 Mean \(After Update via EI\):\s*([0-9.]+)",
    "top3_std_after_ei": r"Top 3 STD \(After Update via EI\):\s*([0-9.]+)",
    "top1_mean_after_ts": r"Top 1 Mean \(After Update via TS\):\s*([0-9.]+)",
    "top1_std_after_ts": r"Top 1 STD \(After Update via TS\):\s*([0-9.]+)",
    "top2_mean_after_ts": r"Top 2 Mean \(After Update via TS\):\s*([0-9.]+)",
    "top2_std_after_ts": r"Top 2 STD \(After Update via TS\):\s*([0-9.]+)",
    "top3_mean_after_ts": r"Top 3 Mean \(After Update via TS\):\s*([0-9.]+)",
    "top3_std_after_ts": r"Top 3 STD \(After Update via TS\):\s*([0-9.]+)",
}


def parse_metrics(text):
    metrics = {}
    for key, pattern in FLOAT_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def load_reference_metrics(model, horizon):
    ref_path = LOG_DIR / REFERENCE_LOGS[model].format(h=horizon)
    if not ref_path.exists():
        return {}
    return parse_metrics(ref_path.read_text(errors="ignore"))


def run_model(model, horizon, python_exec, timeout, extra_env=None):
    script = BASE_DIR / MODEL_SCRIPTS[model]
    out_dir = BASE_DIR / "reproduction_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{model}_H{horizon}.log"

    env = os.environ.copy()
    env["EPISODE_HORIZON"] = str(horizon)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [python_exec, str(script)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    out_path.write_text(proc.stdout)
    return proc.returncode, parse_metrics(proc.stdout), out_path


def flatten_row(model, horizon, run_metrics, ref_metrics):
    row = {"model": model, "horizon": horizon}
    keys = sorted(set(run_metrics) | set(ref_metrics))
    for key in keys:
        row[f"run_{key}"] = run_metrics.get(key, "")
        row[f"reference_{key}"] = ref_metrics.get(key, "")
        if key in run_metrics and key in ref_metrics:
            row[f"delta_{key}"] = run_metrics[key] - ref_metrics[key]
        else:
            row[f"delta_{key}"] = ""
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SCRIPTS),
        choices=list(MODEL_SCRIPTS),
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--env", nargs="*", default=[])
    args = parser.parse_args()

    rows = []
    print(f"Running with Python: {args.python}")
    extra_env = {}
    for item in args.env:
        key, value = item.split("=", 1)
        extra_env[key] = value
    for model in args.models:
        for horizon in args.horizons:
            print(f"\n=== {model} | H={horizon} ===")
            returncode, run_metrics, out_path = run_model(
                model=model,
                horizon=horizon,
                python_exec=args.python,
                timeout=args.timeout,
                extra_env=extra_env,
            )
            ref_metrics = load_reference_metrics(model, horizon)
            print(f"returncode={returncode}")
            print(f"log={out_path}")
            for key in sorted(set(run_metrics) | set(ref_metrics)):
                print(
                    f"{key}: run={run_metrics.get(key, 'NA')} "
                    f"ref={ref_metrics.get(key, 'NA')}"
                )
            rows.append(flatten_row(model, horizon, run_metrics, ref_metrics))

    if rows:
        csv_path = BASE_DIR / "reproduction_outputs" / "summary.csv"
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote summary to {csv_path}")


if __name__ == "__main__":
    main()
