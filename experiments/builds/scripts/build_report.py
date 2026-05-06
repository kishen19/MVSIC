#!/usr/bin/env python3
"""Build-side reporting for the main experiments.

Walks the mirrored build-stats tree at
``experiments/builds/results/indexes/<dataset>/<method>/<build>/build_stats.json``
(populated by ``experiments/builds/scripts/run_builds.sh`` from the per-machine
``results/indexes/`` tree) and emits

    <out_dir>/build_report<tag>.md          per-(dataset, method, build)
                                            markdown -- the canonical
                                            build-stats md
    <out_dir>/_plots/build_time<tag>.pdf    bar plot of build time
    <out_dir>/_plots/index_size<tag>.pdf    bar plot of on-disk index size (MB)

Where ``<tag>`` is empty for the default (every dataset under the indexes
tree), or e.g. ``_beir`` / ``_vidore`` when ``--suite`` is set. This keeps a
crowded all-datasets view next to a clean BEIR-only view in the same
``_plots/`` directory.

The PDFs share the canonical experiments/<stage>/results/_plots/ tree used by
latency / batch / multi_latency / ... ; the build-stats markdown stays at the
results root.

Pointing ``--indexes`` at the mirrored tree (instead of the local
``results/indexes/``) means the report always covers every dataset that's
tracked in the repo, not just the ones whose binaries happen to live on the
current machine.

Usage:
    experiments/builds/scripts/build_report.py                   # all (excl. vidore by default)
    experiments/builds/scripts/build_report.py --suite beir      # BEIR only
    experiments/builds/scripts/build_report.py --suite vidore    # ViDoRe only
    experiments/builds/scripts/build_report.py --suite lotte     # LoTTE only
    experiments/builds/scripts/build_report.py --exclude-suites '' # literally every dataset
    experiments/builds/scripts/build_report.py \
        --datasets nfcorpus,scifact,arguana,scidocs,fiqa
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib numpy to use this script")


_METHOD_ORDER = ["mvivf", "muvera", "vamana", "svh_graph", "fastplaid", "igp"]


# Canonical dataset suites. Mirrors the lists in
# experiments/query_compression/scripts/run_query_compression.sh
# (BEIR_DATASETS / VIDORE_DATASETS) so suite membership stays consistent
# across runners.
_SUITES: dict[str, list[str]] = {
    "beir": [
        "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
        "quora", "nq", "hotpotqa", "nq500k", "msmarco",
    ],
    "vidore": [
        "docvqa", "infovqa", "arxivqa", "tabfquad", "chartqa", "shiftproject",
        "synth_ai", "synth_energy", "synth_gov", "synth_healthcare", "tatdqa",
    ],
    "lotte": [
        "lotte",
    ],
}
_PRETTY = {
    "mvivf": "MVIVF",
    "muvera": "MUVERA",
    "vamana": "MV-Vamana",
    "svh_graph": "SVH Graph",
    "fastplaid": "FastPlaid",
    "igp": "IGP",
}


def _discover_datasets(indexes_root: pathlib.Path) -> list[str]:
    """List every dataset shard under ``indexes_root``.

    ViDoRe puts its shards one level deeper (``indexes/vidore/<sub>``) so
    expand that container into its children rather than treating ``vidore``
    itself as a shard.
    """
    if not indexes_root.exists():
        return []
    out: list[str] = []
    for p in sorted(indexes_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "vidore":
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    out.append(sub.name)
        else:
            out.append(p.name)
    return out


def _walk_stats(indexes_root: pathlib.Path,
                datasets: list[str] | None) -> pd.DataFrame:
    rows = []
    if datasets is None:
        datasets = _discover_datasets(indexes_root)
        if not datasets:
            return pd.DataFrame()
    for ds in datasets:
        ds_dir = indexes_root / ds
        # vidore puts datasets one level deeper.
        if (indexes_root / "vidore" / ds).exists():
            ds_dir = indexes_root / "vidore" / ds
        if not ds_dir.exists():
            continue
        for stats_path in ds_dir.rglob("build_stats.json"):
            try:
                with open(stats_path) as f:
                    s = json.load(f)
            except Exception as e:
                print(f"[skip {stats_path}] {e}")
                continue
            rel = stats_path.relative_to(ds_dir)
            parts = rel.parts  # <method>/<build>/build_stats.json
            if len(parts) < 3:
                continue
            row = {
                "dataset": ds,
                "method": parts[0],
                "build_name": parts[1],
                "build_time_sec": s.get("build_time_sec"),
                "index_size_mb": s.get("index_size_mb"),
                "built_at": s.get("built_at"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _bar_plot(df: pd.DataFrame, value_col: str, ylabel: str,
              out_path: pathlib.Path, log: bool = False) -> None:
    if df.empty or value_col not in df.columns or df[value_col].dropna().empty:
        return
    datasets = sorted(df["dataset"].unique())
    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    if not datasets or not methods:
        return
    cmap = plt.colormaps.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(datasets) + 2), 4.5))
    width = 0.8 / max(1, len(methods))
    x_centers = np.arange(len(datasets))
    for mi, method in enumerate(methods):
        heights = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["method"] == method)]
            v = sub[value_col].dropna()
            heights.append(float(v.iloc[0]) if not v.empty else np.nan)
        offsets = (mi - (len(methods) - 1) / 2.0) * width
        ax.bar(x_centers + offsets, heights, width=width,
               label=_PRETTY.get(method, method), color=cmap(mi % 10))
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _markdown_table(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    if df.empty:
        out_path.write_text("# Build report\n\n_No build_stats.json files found._\n")
        print(f"  wrote {out_path} (empty)")
        return

    cols = [
        "dataset", "method", "build_name",
        "build_time_sec", "index_size_mb",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    fmt_map = {
        "build_time_sec": lambda v: "" if pd.isna(v) else f"{v:.1f}",
        "index_size_mb":  lambda v: "" if pd.isna(v) else f"{v:.2f}",
    }
    for c, fn in fmt_map.items():
        if c in df.columns:
            df[c] = df[c].map(fn)
    df = df.sort_values(["dataset", "method", "build_name"])

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                     for _, r in df.iterrows())
    md = "# Build report\n\n" + "\n".join([header, sep, body]) + "\n"
    out_path.write_text(md)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--indexes",
        type=pathlib.Path,
        default=pathlib.Path("experiments/builds/results/indexes"),
        help="Root of the build-stats tree. Default points at the mirrored, "
             "tracked tree under experiments/builds/results/indexes/ (covers "
             "every dataset). Pass results/indexes/ to use the per-machine "
             "real-binary tree instead.",
    )
    p.add_argument(
        "--suite",
        choices=("all", *_SUITES.keys()),
        default="all",
        help="Dataset suite preset (mutually exclusive with --datasets). "
             "'all' = every dataset under --indexes minus --exclude-suites; "
             "'beir' = nfcorpus, scifact, arguana, scidocs, fiqa, quora, nq, "
             "hotpotqa, nq500k, msmarco; 'vidore' = the ColPali ViDoRe shards; "
             "'lotte' = the LoTTE corpus. Output filenames are tagged with the "
             "suite (e.g. build_time_beir.pdf) so multiple suites coexist in "
             "the same _plots/ directory.",
    )
    p.add_argument(
        "--exclude-suites",
        default="vidore",
        help="Comma-separated list of suites to drop from the --suite all "
             "view (does nothing for --suite <name> or --datasets). Default: "
             "'vidore' so the all-datasets bar chart stays legible. Pass "
             "'' (empty) to keep every dataset, or e.g. 'vidore,lotte' to "
             "drop both. Use --suite vidore to plot just vidore.",
    )
    p.add_argument("--datasets", default=None,
                   help="Comma-separated dataset names; overrides --suite.")
    p.add_argument(
        "--tag",
        default=None,
        help="Custom filename suffix (default: derived from --suite). "
             "Pass an empty string to skip suffixing entirely.",
    )
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="Default: experiments/builds/results/")
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else pathlib.Path("experiments/builds/results")
    if args.datasets is not None:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif args.suite != "all":
        datasets = list(_SUITES[args.suite])
    else:
        # --suite all: discover everything under --indexes, then subtract any
        # datasets belonging to suites named in --exclude-suites. ViDoRe is
        # excluded by default to keep the bar chart legible; --suite vidore
        # remains available for an explicit vidore-only plot.
        datasets = _discover_datasets(args.indexes)
        excluded: set[str] = set()
        for s in (args.exclude_suites or "").split(","):
            s = s.strip()
            if not s:
                continue
            if s not in _SUITES:
                print(f"[warn] --exclude-suites: unknown suite '{s}' "
                      f"(known: {sorted(_SUITES)}); ignoring.",
                      file=sys.stderr)
                continue
            excluded.update(_SUITES[s])
        if excluded:
            kept = [d for d in datasets if d not in excluded]
            dropped = [d for d in datasets if d in excluded]
            if dropped:
                print(f"[info] --suite all: dropping {len(dropped)} dataset(s) "
                      f"via --exclude-suites='{args.exclude_suites}': "
                      f"{', '.join(dropped)}")
            datasets = kept
        if not datasets:
            datasets = None  # nothing to filter; let _walk_stats discover

    df = _walk_stats(args.indexes, datasets)
    if df.empty:
        print(f"[warn] no build_stats.json under {args.indexes}; nothing to report.")
        return 0

    if args.tag is not None:
        tag = args.tag
    elif args.suite != "all":
        tag = f"_{args.suite}"
    else:
        tag = ""
    if tag and not tag.startswith("_"):
        tag = "_" + tag

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _markdown_table(df, out_dir / f"build_report{tag}.md")
    _bar_plot(df, "build_time_sec", "Build time (s)",
              plots_dir / f"build_time{tag}.pdf", log=True)
    _bar_plot(df, "index_size_mb", "Index size on disk (MB)",
              plots_dir / f"index_size{tag}.pdf", log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
