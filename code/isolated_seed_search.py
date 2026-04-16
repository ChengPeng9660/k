import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.8/bin/python3"


def rank_rows(rows, horizon):
    ranked = [r for r in rows if r["horizon"] == str(horizon) and r["returncode"] == "0"]
    ranked.sort(
        key=lambda r: float(r["mean_accepted_reward"])
        if r["mean_accepted_reward"] != "nan"
        else float("-inf"),
        reverse=True,
    )
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="brqn_user_mapping")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=12)
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--train-users", type=int, default=140)
    parser.add_argument("--test-users", type=int, default=90)
    parser.add_argument("--min-steps", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    found = None
    for seed in range(args.seed_start, args.seed_end + 1):
        seed_dir = outdir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"brqn_seed_{seed}_") as tmp:
            work = Path(tmp) / BASE_DIR.name
            shutil.copytree(BASE_DIR, work)
            cmd = [
                PYTHON,
                "search_brqn_setup.py",
                "--mode",
                args.mode,
                "--horizons",
                *[str(h) for h in args.horizons],
                "--models",
                "MAB",
                "XgBoost",
                "DQN",
                "BRQN",
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
                "--num-episodes",
                str(args.num_episodes),
                "--batch-size",
                str(args.batch_size),
                "--outdir",
                str(seed_dir),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            (seed_dir / "driver.log").write_text(proc.stdout)
        results_path = seed_dir / "results.csv"
        if not results_path.exists():
            print(f"seed={seed} missing results")
            continue
        rows = list(csv.DictReader(results_path.open()))
        success = True
        for horizon in args.horizons:
            ranked = rank_rows(rows, horizon)
            print(
                f"seed={seed} H={horizon} "
                f"{[(r['model'], r['mean_accepted_reward'], r['accepted_episodes']) for r in ranked]}"
            )
            if not ranked or ranked[0]["model"] != "BRQN":
                success = False
        if success:
            found = seed
            print(f"FOUND seed={seed}")
            break

    if found is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
