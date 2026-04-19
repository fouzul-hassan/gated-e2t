"""
fetch_results.py
----------------
Fetch pre-computed mean test metrics from a W&B run summary.

W&B logs per-subject scores during training; at the end of each test step it
also logs cross-subject mean values under keys like:
    test/mean_BLEU1@MTV
    test/mean_ROUGE1@RAW
    test/mean_retrieval_acc_top01
    test/mean_relation_cls_acc_top01
    test/mean_corpus_cls_acc

This script reads those pre-aggregated means directly from `run.summary`
(the W&B summary dict), which is the authoritative source.

Usage
-----
  python fetch_results.py --run "fouzul-hassan3-iit-sri-lanka/glim/runs/p67g6dls"

  # Compare multiple runs side-by-side
  python fetch_results.py \\
      --run "entity/project/runs/abc123" \\
      --run "entity/project/runs/def456" \\
      --names "Model-V1" "Model-V2"
"""

import argparse
import warnings

import pandas as pd
import wandb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Mapping: display name  ->  W&B summary key (pre-computed mean over subjects)
#
# W&B stores these under run.summary after each test epoch, e.g.:
#   test/mean_BLEU1@MTV         BLEU-1 (generated, multi-task-vocabulary)
#   test/mean_BLEU2@MTV         BLEU-2
#   test/mean_ROUGE1@RAW        ROUGE-1 F-measure (raw, generated)
#   test/mean_retrieval_acc_top01
#   test/mean_retrieval_acc_top05
#   test/mean_relation_cls_acc_top01
#   test/mean_relation_cls_acc_top03
#   test/mean_corpus_cls_acc
# ---------------------------------------------------------------------------
SUMMARY_KEYS = {
    # Generation
    "BLEU-1":           "test/mean_BLEU1@MTV",
    "BLEU-2":           "test/mean_BLEU2@MTV",
    "ROUGE-1 RAW":      "test/mean_ROUGE1@RAW",
    # Retrieval
    "ACC-1 Retrieval":  "test/mean_retrieval_acc_top01",
    "ACC-5 Retrieval":  "test/mean_retrieval_acc_top05",
    # Relation Classification
    "ACC-1 Relation":   "test/mean_relation_cls_acc_top01",
    "ACC-3 Relation":   "test/mean_relation_cls_acc_top03",
    # Corpus Classification
    "ACC Corpus":       "test/mean_corpus_cls_acc",
}

# Which group each metric belongs to
METRIC_GROUP = {
    "BLEU-1":           "Generation",
    "BLEU-2":           "Generation",
    "ROUGE-1 RAW":      "Generation",
    "ACC-1 Retrieval":  "Retrieval",
    "ACC-5 Retrieval":  "Retrieval",
    "ACC-1 Relation":   "Classification",
    "ACC-3 Relation":   "Classification",
    "ACC Corpus":       "Classification",
}

# Display order
GROUP_ORDER = ["Generation", "Retrieval", "Classification"]
METRIC_ORDER = [m for g in GROUP_ORDER for m, mg in METRIC_GROUP.items() if mg == g]


def _extract_value(v) -> float:
    """
    W&B summary values can be plain floats OR SummarySubDict objects like
    {'max': 0.123}.  SummarySubDict is NOT a plain dict, so we use try/except
    key-access which works for both types.
    """
    # Try subscript access for dict / SummarySubDict  {'max': value}
    try:
        return float(v["max"])
    except (KeyError, TypeError):
        pass
    # Plain numeric value
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fetch_run_metrics(run_path: str) -> dict[str, float]:
    """
    Read pre-computed cross-subject mean scores directly from W&B run.summary.

    W&B logs these aggregated means at the end of each test step under keys
    like  test/mean_BLEU1@MTV,  test/mean_retrieval_acc_top01, etc.

    Returns a dict: display_name -> float value (NaN if key not present).
    """
    api = wandb.Api()
    run = api.run(run_path)
    summary = run.summary  # dict-like object

    results = {}
    missing = []
    for metric_name, key in SUMMARY_KEYS.items():
        if key in summary:
            results[metric_name] = _extract_value(summary[key])
        else:
            results[metric_name] = float("nan")
            missing.append(key)

    if missing:
        print(f"  [warn] keys not found in summary: {missing}")

    return results


def build_table(run_paths: list[str], names: list[str]) -> pd.DataFrame:
    """Fetch metrics for every run and stack them into a single DataFrame."""
    rows = []
    for path, name in zip(run_paths, names):
        print(f"  Fetching: {path}  ({name})")
        metrics = fetch_run_metrics(path)
        metrics["Model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Model")
    return df[METRIC_ORDER]          # enforce display order


def print_table(df: pd.DataFrame):
    """Pretty-print grouped results table (ASCII-safe for Windows consoles)."""
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)

    sep = "=" * 80
    print("\n" + sep)
    print("  W&B Test Results")
    print(sep)

    for group in GROUP_ORDER:
        group_metrics = [m for m in METRIC_ORDER if METRIC_GROUP[m] == group]
        sub = df[group_metrics]

        print(f"\n-- {group} --")
        print(sub.to_string())

    print("\n" + sep)

    # Also print a flat table for easy copy-paste
    print("\n[Full table]\n")
    print(df.to_string())
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch & aggregate test metrics from one or more W&B runs."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        metavar="ENTITY/PROJECT/runs/RUN_ID",
        help=(
            "W&B run path, e.g. 'fouzul-hassan3-iit-sri-lanka/glim/runs/p67g6dls'. "
            "Repeat the flag to compare multiple runs side-by-side."
        ),
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Display names for each run (same order as --run). "
            "Defaults to the run IDs."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Optional path to save the results as a CSV file.",
    )
    args = parser.parse_args()

    run_paths = args.runs
    names = args.names if args.names else [p.split("/")[-1] for p in run_paths]

    if len(names) != len(run_paths):
        parser.error(
            f"Number of --names ({len(names)}) must match number of --run ({len(run_paths)}) arguments."
        )

    df = build_table(run_paths, names)
    print_table(df)

    if args.csv:
        df.to_csv(args.csv)
        print(f"Results saved to: {args.csv}")


if __name__ == "__main__":
    main()
