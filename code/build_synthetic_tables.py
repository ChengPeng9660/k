import argparse
import csv
import math
import statistics
from pathlib import Path


BENCHMARK = ["MAB", "XgBoost", "DQN", "DRQN", "BRQN"]
ABLATION = ["DNN_BayesianUpdate", "RNN_BayesianUpdate", "BDQN", "BRQN"]


def to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def mean_std(values):
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan, math.nan
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def fmt(x):
    if math.isnan(x):
        return "NA"
    return f"{x:.2f}"


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in [
                "mean_all_reward",
                "mean_accepted_reward",
                "accepted_episodes",
                "total_candidate_episodes",
                "acceptance_rate",
            ]:
                row[key] = to_float(row[key])
            rows.append(row)
    return rows


def summarize(rows, models):
    out = []
    for model in models:
        sub = [r for r in rows if r["model"] == model and int(r["returncode"]) == 0]
        mean_all, std_all = mean_std([r["mean_all_reward"] for r in sub])
        mean_acc, std_acc = mean_std([r["mean_accepted_reward"] for r in sub])
        mean_count, std_count = mean_std([r["accepted_episodes"] for r in sub])
        mean_total, _ = mean_std([r["total_candidate_episodes"] for r in sub])
        mean_rate, std_rate = mean_std([r["acceptance_rate"] for r in sub])
        out.append(
            {
                "model": model,
                "accepted_reward": mean_acc,
                "accepted_reward_std": std_acc,
                "all_reward": mean_all,
                "all_reward_std": std_all,
                "accepted_count": mean_count,
                "accepted_count_std": std_count,
                "total_count": mean_total,
                "acceptance_rate": mean_rate,
                "acceptance_rate_std": std_rate,
            }
        )
    out.sort(key=lambda x: (math.isnan(x["accepted_reward"]), -x["accepted_reward"] if not math.isnan(x["accepted_reward"]) else 0))
    return out


def markdown_table(title, rows):
    lines = [f"## {title}", "", "| Model | Accepted reward | All reward | Accepted / Total | Acceptance rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        acc = f"{fmt(row['accepted_reward'])} +- {fmt(row['accepted_reward_std'])}"
        all_r = f"{fmt(row['all_reward'])} +- {fmt(row['all_reward_std'])}"
        counts = f"{fmt(row['accepted_count'])} / {fmt(row['total_count'])}"
        rate = f"{fmt(row['acceptance_rate'])} +- {fmt(row['acceptance_rate_std'])}"
        lines.append(f"| {row['model']} | {acc} | {all_r} | {counts} | {rate} |")
    return "\n".join(lines)


def latex_table(caption, label, rows):
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Model & Accepted reward & All reward & Accepted / Total & Acceptance rate \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & "
            f"{fmt(row['accepted_reward'])} $\\pm$ {fmt(row['accepted_reward_std'])} & "
            f"{fmt(row['all_reward'])} $\\pm$ {fmt(row['all_reward_std'])} & "
            f"{fmt(row['accepted_count'])} / {fmt(row['total_count'])} & "
            f"{fmt(row['acceptance_rate'])} $\\pm$ {fmt(row['acceptance_rate_std'])} \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="reproduction_outputs/multiseed_h1/multiseed_results.csv")
    parser.add_argument("--outdir", default="reproduction_outputs/multiseed_h1")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bench = summarize(rows, BENCHMARK)
    ablation = summarize(rows, ABLATION)

    md = [
        "# Synthetic H=1 Multi-Seed Tables",
        "",
        "Synthetic data, H=1, debug version. Primary metric: accepted reward.",
        "",
        markdown_table("Benchmark Comparison", bench),
        "",
        markdown_table("Ablation Study", ablation),
        "",
        "## Note",
        "",
        "- `XgBoost` here may use the sklearn fallback on this machine when native xgboost cannot load `libomp.dylib`.",
    ]
    (outdir / "synthetic_multiseed_tables.md").write_text("\n".join(md))

    tex = [
        latex_table("Synthetic benchmark comparison under H=1 (debug version).", "tab:synthetic_benchmark_h1", bench),
        "",
        latex_table("Synthetic ablation study under H=1 (debug version).", "tab:synthetic_ablation_h1", ablation),
    ]
    (outdir / "synthetic_multiseed_tables.tex").write_text("\n\n".join(tex))


if __name__ == "__main__":
    main()
