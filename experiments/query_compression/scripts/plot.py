#!/usr/bin/env python3
"""Plot recall vs τ and average compressed query size from sweep CSVs.

Expected columns (k=10 example):
  tau, recall_1_10, recall_10_10, avg_compressed_vectors

Outputs go to the shared experiments/query_compression/results/_plots/
directory (matching the latency / batch / multi_latency / ... convention).
File names embed the dataset, e.g. recall_wards_<ds>.pdf and
recall_ball_carving_<ds>.pdf.

Usage:
  scripts/plot.py --dataset arguana \\
    --results-dir experiments/query_compression/results/arguana
  # or override the destination explicitly:
  scripts/plot.py --dataset arguana \\
    --results-dir experiments/query_compression/results/arguana \\
    --out-dir experiments/query_compression/results/_plots
"""

from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib")

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _plot_csv(csv_path: pathlib.Path, out_pdf: pathlib.Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if "tau" not in df.columns:
        raise SystemExit(f"{csv_path}: missing 'tau' column")

    df = df.sort_values("tau").reset_index(drop=True)
    tau = df["tau"]

    recall_cols = [c for c in df.columns if c.startswith("recall_")]
    has_avg = "avg_compressed_vectors" in df.columns

    fig, ax_r = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    ax_q = ax_r.twinx() if has_avg else None

    cmap = plt.colormaps.get_cmap("tab10")

    def recall_label(col: str) -> str:
        parts = col.split("_")
        if len(parts) == 3 and parts[0] == "recall":
            return f"Recall-${parts[1]}$@${parts[2]}$"
        return col.replace("_", " ")

    legend_loc = "lower right" if "ball_carving" in csv_path.stem else "best"

    for i, col in enumerate(recall_cols):
        ax_r.plot(
            tau,
            df[col],
            color=cmap(i % 10),
            lw=2.8,
            zorder=4,
            label=recall_label(col),
        )

    ax_r.set_xlabel(r"Threshold $\tau$")
    ax_r.set_ylabel("Recall")
    ax_r.set_ylim(0.0, 1.02)
    ax_r.grid(True, which="both", alpha=0.3)
    ax_r.set_axisbelow(True)
    ax_r.spines["top"].set_visible(False)

    lines_r, labels_r = ax_r.get_legend_handles_labels()

    if has_avg and ax_q is not None:
        qcol = cmap(min(len(recall_cols), 9))
        ax_q.plot(
            tau,
            df["avg_compressed_vectors"],
            color=qcol,
            lw=2.8,
            linestyle="--",
            zorder=3,
            label="Avg Query Size",
        )
        ax_q.set_ylabel("Avg Query Size")
        ax_q.tick_params(axis="y", colors=qcol)
        ax_q.spines["top"].set_visible(False)

        lines_q, labels_q = ax_q.get_legend_handles_labels()
        ax_r.legend(
            lines_r + lines_q,
            labels_r + labels_q,
            loc=legend_loc,
            ncol=1,
            fontsize=13,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
        )
    else:
        ax_r.legend(
            lines_r,
            labels_r,
            loc=legend_loc,
            ncol=1,
            fontsize=13,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
        )

    ax_r.set_title(title)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", type=pathlib.Path, required=True,
                    help="Per-dataset results dir, e.g. "
                         "experiments/query_compression/results/<ds>/.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None,
                    help="Default: <results-dir>/../_plots (canonical _plots/ "
                         "tree shared across experiments).")
    args = ap.parse_args()

    ds = args.dataset
    rd = args.results_dir
    od = args.out_dir if args.out_dir is not None else rd.parent / "_plots"
    od.mkdir(parents=True, exist_ok=True)

    pairs = [
        (rd / "ball_carving.csv", od / f"recall_ball_carving_{ds}.pdf", f"{ds} - Ball Carving"),
        (rd / "wards.csv", od / f"recall_wards_{ds}.pdf", f"{ds} - Ward's Method"),
    ]
    for csv_path, pdf_path, title in pairs:
        if not csv_path.is_file():
            print(f"[plot] skip (missing): {csv_path}")
            continue
        _plot_csv(csv_path, pdf_path, title)


if __name__ == "__main__":
    main()
