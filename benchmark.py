"""Run the paired benchmark and export its learning curves and acquisition trace."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from active_learning import add_intercept, classification_metrics, learn
from digits_data import DEFAULT_ARCHIVE, MEMBERS, fetch_dataset, read_digits


DEFAULT_SEEDS = [17, 29, 43, 71, 113]
DEFAULT_BUDGETS = [40, 80, 120, 160, 200, 240]
STRATEGIES = ("random", "entropy")


def run_experiment(training, testing, seeds, budgets):
    train_x, train_y = training
    test_x, test_y = testing
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Specify unique nonnegative integer seeds.")
    if not budgets or budgets != sorted(set(budgets)) or min(budgets) < 1 or max(budgets) > len(train_y):
        raise ValueError("Budgets must increase strictly and fit inside the training pool.")
    metrics, acquisitions = [], []
    test_design = add_intercept(test_x)
    for seed in seeds:
        initial_seed, acquisition_seed = np.random.SeedSequence(seed).spawn(2)
        initial = np.random.default_rng(initial_seed).choice(len(train_y), budgets[0], replace=False)
        for strategy in STRATEGIES:
            checkpoints = learn(train_x, train_y.__getitem__, initial, budgets, strategy,
                                np.random.default_rng(acquisition_seed))
            for point in checkpoints:
                predicted = (test_design @ point.weights).argmax(axis=1)
                accuracy, f1 = classification_metrics(test_y, predicted)
                metrics.append(dict(seed=seed, strategy=strategy, labels=point.budget,
                                    accuracy=accuracy, macro_f1=f1))
                for row_id in point.acquired:
                    acquisitions.append(dict(seed=seed, strategy=strategy, labels=point.budget,
                                             train_row=int(row_id), digit=int(train_y[row_id])))
    return metrics, acquisitions


def summarize(metrics, seeds, budgets):
    by_key = {(row["seed"], row["strategy"], row["labels"]): row for row in metrics}
    summary = []
    for budget in budgets:
        row = {"labels": budget}
        for metric in ("accuracy", "macro_f1"):
            random = [by_key[s, "random", budget][metric] for s in seeds]
            entropy = [by_key[s, "entropy", budget][metric] for s in seeds]
            diff = [e - r for e, r in zip(entropy, random)]
            row.update({f"random_{metric}": mean(random), f"entropy_{metric}": mean(entropy),
                        f"paired_{metric}_gain": mean(diff),
                        f"paired_{metric}_gain_sd": stdev(diff) if len(diff) > 1 else None,
                        f"entropy_{metric}_wins": sum(d > 0 for d in diff)})
        summary.append(row)
    return summary


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Path to the original UCI ZIP.")
    parser.add_argument("--fetch", action="store_true", help="Download a missing archive from UCI.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--out", type=Path, default=Path("run"))
    args = parser.parse_args()
    if args.fetch:
        fetch_dataset(args.archive)
    if not args.archive.exists():
        parser.error("Dataset missing. Add --fetch or supply --archive /path/to/optdigits.zip.")
    training, testing = read_digits(args.archive)
    try:
        metrics, acquisitions = run_experiment(training, testing, args.seeds, args.budgets)
    except ValueError as error:
        parser.error(str(error))
    summary = summarize(metrics, args.seeds, args.budgets)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "metrics.csv", metrics)
    write_rows(args.out / "queries.csv", acquisitions)
    write_rows(args.out / "summary.csv", summary)
    manifest = dict(seeds=args.seeds, budgets=args.budgets, numpy_version=np.__version__,
                    sha256={name: info[1] for name, info in MEMBERS.items()})
    (args.out / "settings.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Labels     Random accuracy    Entropy accuracy    Paired gain")
    for row in summary:
        print(f"{row['labels']:6d} {row['random_accuracy']:18.2%} {row['entropy_accuracy']:19.2%} "
              f"{row['paired_accuracy_gain'] * 100:+11.2f} pp")


if __name__ == "__main__":
    main()
