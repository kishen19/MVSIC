#!/usr/bin/env python3
"""Plot recall vs τ and average compressed query size from sweep CSVs.

Expected columns (k=10 example):
  tau, recall_1_10, recall_10_10, avg_compressed_vectors

Usage:
  scripts/plot.py --dataset arguana \\
    --results-dir experiments/query_compression/results/arguana \\
    --out-dir experiments/query_compression/results/arguana/plots
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


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
            "axes.facecolor": "#fafaf8",
            "axes.edgecolor": "#d6d3cd",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#292524",
            "axes.titlecolor": "#1c1917",
            "xtick.color": "#57534e",
            "ytick.color": "#57534e",
            "grid.color": "#e7e5e4",
            "grid.linewidth": 0.8,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#e7e5e4",
        }
    )


def _plot_csv(csv_path: pathlib.Path, out_pdf: pathlib.Path, title: str) -> None:
    _apply_plot_style()
    df = pd.read_csv(csv_path)
    if "tau" not in df.columns:
        raise SystemExit(f"{csv_path}: missing 'tau' column")

    df = df.sort_values("tau").reset_index(drop=True)
    tau = df["tau"]

    recall_cols = [c for c in df.columns if c.startswith("recall_")]
    has_avg = "avg_compressed_vectors" in df.columns

    fig = plt.figure(figsize=(9.2, 5.4))
    fig.patch.set_facecolor("#f4f2ef")
    ax_r = fig.add_subplot(111, facecolor="#fafaf8")
    ax_q = ax_r.twinx() if has_avg else None

    recall_colors = ("#1d4ed8", "#7c3aed")
    recall_markers = ("o", "s")

    for i, col in enumerate(recall_cols):
        ax_r.plot(
            tau,
            df[col],
            marker=recall_markers[i % len(recall_markers)],
            color=recall_colors[i % len(recall_colors)],
            lw=2.35,
            ms=6,
            markeredgewidth=1.0,
            markeredgecolor="white",
            zorder=4,
            label=col.replace("_", " "),
        )

    ax_r.set_xlabel(r"Threshold $\tau$", fontsize=11.5, labelpad=8)
    ax_r.set_ylabel("Recall", fontsize=11.5, color="#1c1917", labelpad=10)
    ax_r.tick_params(axis="y", colors="#44403c")
    ax_r.grid(True, alpha=0.85, linestyle="-", linewidth=0.7)
    ax_r.set_axisbelow(True)

    lines_r, labels_r = ax_r.get_legend_handles_labels()

    if has_avg and ax_q is not None:
        qcol = "#c2410c"
        ax_q.plot(
            tau,
            df["avg_compressed_vectors"],
            color=qcol,
            lw=2.5,
            linestyle=(0, (6, 4)),
            marker="D",
            ms=5.5,
            markeredgewidth=1.0,
            markeredgecolor="white",
            zorder=3,
            label="Avg compressed size",
        )
        ax_q.fill_between(
            tau,
            df["avg_compressed_vectors"],
            alpha=0.12,
            color=qcol,
            zorder=1,
        )
        ax_q.set_ylabel(
            "Avg vectors per query\n(after compression)",
            fontsize=11,
            color=qcol,
            labelpad=12,
        )
        ax_q.tick_params(axis="y", colors=qcol)
        ax_q.spines["top"].set_visible(False)
        ax_q.spines["right"].set_color(qcol)
        ax_q.spines["right"].set_linewidth(1.2)

        lines_q, labels_q = ax_q.get_legend_handles_labels()
        ax_r.legend(
            lines_r + lines_q,
            labels_r + labels_q,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=min(3, len(lines_r) + len(lines_q)),
            fontsize=9,
            frameon=True,
            fancybox=False,
            shadow=False,
        )
    else:
        ax_r.legend(
            lines_r,
            labels_r,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=min(2, len(lines_r)),
            fontsize=9,
            frameon=True,
        )

    ax_r.spines["top"].set_visible(False)

    fig.suptitle(
        title,
        fontsize=13.5,
        fontweight="600",
        y=1.02,
        color="#1c1917",
    )

    fig.subplots_adjust(left=0.1, right=0.88, top=0.88, bottom=0.28 if has_avg else 0.22)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    ds = args.dataset
    rd = args.results_dir
    od = args.out_dir
    od.mkdir(parents=True, exist_ok=True)

    pairs = [
        (rd / "ball_carving.csv", od / f"recall_ball_carving_{ds}.pdf", f"{ds} — ball carving"),
        (rd / "wards.csv", od / f"recall_wards_{ds}.pdf", f"{ds} — Ward linkage"),
    ]
    for csv_path, pdf_path, title in pairs:
        if not csv_path.is_file():
            print(f"[plot] skip (missing): {csv_path}")
            continue
        _plot_csv(csv_path, pdf_path, title)


if __name__ == "__main__":
    main()
