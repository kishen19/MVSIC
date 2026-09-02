#!/usr/bin/env python3
"""Optimizations-ladder plots (styling aligned with experiments/builds/scripts/plot_stage.py).

Per dataset under ``experiments/optimizations/results/<dataset>/``:

  1. ``<dataset>_pareto.pdf`` — Recall (k@k) vs latency ms (1000/QPS_seq), Pareto
     curves only (no scatter "ghost" points).

  1b. ``<ds1>_<ds2>..._paper_pareto.pdf`` (once, not per dataset) — the same
      Pareto panels side by side, one column per loaded dataset, with a
      single shared stage legend on top. Written whenever >=2 datasets load
      successfully in one invocation.

  2. ``<dataset>_ladder_r{90,95,99}.pdf`` — one latency ladder per recall regime.
  3. ``<dataset>_ladder_combined.pdf`` — row of vertical-bar panels per recall regime
     (independent y-scales), legend for stages, latency (ms) on bar tops.

  3. ``<dataset>_speedup_table.md`` — same numbers in tabular form.

Usage:
    scripts/plot.py
    scripts/plot.py --datasets arguana,nq
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Optional

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib numpy to use this script")


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
_BUILD_SCRIPTS = REPO_ROOT / "experiments" / "builds" / "scripts"
if str(_BUILD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BUILD_SCRIPTS))

import plot_stage as ps  # noqa: E402


# Match plot_stage.py typography.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

_QPS = "QPS_seq"
_RECALL_COL = "recall_k_k"
_XLIM = (ps._XLIM_LEFT_RECALL_K_K, ps._XLIM_RIGHT_RECALL)

_STAGE_ORDER = [
    "s0_baseline",
    "s1_leaf_tq1bit",
    "s2_tree_compress",
    "s3_query_compress",
    "s4_rerank_tq8bit",
]
# Legend / axis labels (Rand-* naming matches overretrieve plots).
_STAGE_LABELS = {
    "s0_baseline":       "Raw MaxSim",
    "s1_leaf_tq1bit":    "+ Leaf Quant (Rand-1bit)",
    "s2_tree_compress":  "+ Tree Quant (TQ-4bit)",
    "s3_query_compress": "+ Query Compress (Ward's)",
    "s4_rerank_tq8bit":  "+ Rerank Quant (Rand-8bit)",
}
_RECALL_THRESHOLDS = (0.90, 0.95, 0.99)
# Short labels under each recall cluster (plot_stage uses Recall-$k$@$k$ on Pareto x-axis).
_RECALL_CLUSTER_LABELS = {
    0.90: r"$\geq$90%",
    0.95: r"$\geq$95%",
    0.99: r"$\geq$99%",
}
_BAR_SIDE_PAD = 0.55       # center the bar group in the axes
_BAR_WIDTH = 1.0           # touching vertical bars


def _ladder_title(dataset: str, k_val: int, recall: float | None = None) -> str:
    """e.g. arguana: Latency (ms) at $\\geq$90% Recall-$10$@$10$"""
    rx = ps._recall_xlabel("k", k_val)
    if recall is not None:
        rlab = _RECALL_CLUSTER_LABELS.get(recall, f"{recall:.0%}")
        return f"{dataset}: Latency (ms) at {rlab} {rx}"
    return f"{dataset}: Latency (ms) at {rx}"


def _ladder_panel_title(recall: float) -> str:
    """Short panel header (GEM-style): $\\geq$90%"""
    return _RECALL_CLUSTER_LABELS.get(recall, f"{recall:.0%}")


# Cool → warm along the ladder (distinct from tab10 method colors).
def _stage_color(i: int, n: int) -> tuple:
    cmap = plt.colormaps.get_cmap("viridis")
    t = 0.12 + 0.76 * (i / max(1, n - 1))
    return cmap(t)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "qps_seq" in df.columns and "QPS_seq" not in df.columns:
        rename["qps_seq"] = "QPS_seq"
    if "qps_par" in df.columns and "QPS_par" not in df.columns:
        rename["qps_par"] = "QPS_par"
    return df.rename(columns=rename) if rename else df


def _load_dataset(results_root: pathlib.Path, dataset: str) -> pd.DataFrame:
    base = results_root / dataset
    if not base.exists():
        raise SystemExit(f"no results directory: {base}")
    files = sorted(base.rglob("latency_*.csv"))
    if not files:
        raise SystemExit(f"no latency_*.csv files under {base}")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        df = _normalize_columns(df)
        rel = f.relative_to(base)
        parts = rel.parts
        if len(parts) < 5:
            continue
        df["index_name"] = parts[0]
        df["build"] = parts[1]
        df["variant"] = parts[2]
        df["search"] = parts[3]
        rows.append(df)
    if not rows:
        raise SystemExit(f"all latency_*.csv files under {base} were empty")
    return pd.concat(rows, ignore_index=True)


def _latency_at_recall(
    fx: np.ndarray, lat_ms: np.ndarray, target: float,
) -> Optional[float]:
    """Interpolate latency (ms) along the Pareto front at ``target`` recall."""
    if fx.size == 0:
        return None
    if target <= float(fx.min()):
        return float(lat_ms[np.argmin(fx)])
    if target > float(fx.max()):
        return None
    i = int(np.searchsorted(fx, target, side="left"))
    i = max(1, min(i, fx.size - 1))
    x0, x1 = float(fx[i - 1]), float(fx[i])
    y0, y1 = float(lat_ms[i - 1]), float(lat_ms[i])
    if x1 == x0:
        return min(y0, y1)
    t = (target - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def _stage_pareto_xy(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pareto front in recall vs latency ms (envelope over nprobes × num_rerank)."""
    sub = sub.dropna(subset=[_RECALL_COL, _QPS])
    if sub.empty:
        return np.array([]), np.array([])
    xs = sub[_RECALL_COL].to_numpy(dtype=float)
    qv = sub[_QPS].to_numpy(dtype=float)
    return ps._pareto_curve_recall_latency_ms(xs, qv)


def _plot_pareto_panel(
    ax,
    df: pd.DataFrame,
    dataset: str,
    *,
    show_ylabel: bool = True,
    show_legend: bool = True,
    show_title: bool = False,
) -> list[str]:
    """Draw the per-stage viridis Pareto curves for one dataset onto ``ax``.

    Shared by the single-dataset plot and the multi-dataset side-by-side
    plot so both stay pixel-identical for the stages they have in common.
    Returns the stage keys actually plotted (empty if none).
    """
    stages = [s for s in _STAGE_ORDER if s in df["variant"].unique()]
    if not stages:
        return []

    k_val = int(df["k"].dropna().iloc[0]) if "k" in df.columns and not df["k"].dropna().empty else 10
    n = len(stages)

    plotted_min_recall: float | None = None
    plotted_max_recall: float | None = None
    for i, stage in enumerate(stages):
        sub = df[df["variant"] == stage]
        fx, fy = _stage_pareto_xy(sub)
        if fx.size == 0:
            continue
        color = _stage_color(i, n)
        ax.plot(
            fx,
            fy,
            marker="o",
            markersize=5.4,
            linewidth=2.8,
            linestyle="-",
            color=color,
            label=_STAGE_LABELS.get(stage, stage),
            zorder=3,
        )
        xmin = float(np.nanmin(fx))
        xmax = float(np.nanmax(fx))
        plotted_min_recall = xmin if plotted_min_recall is None else min(plotted_min_recall, xmin)
        plotted_max_recall = xmax if plotted_max_recall is None else max(plotted_max_recall, xmax)

    ax.set_xlabel(ps._recall_xlabel("k", k_val))
    if show_ylabel:
        ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    # Default: high-recall zoom (0.82 .. 1.002). If a stage's Pareto tops out
    # below that (e.g. hotpotqa's raw baseline maxes near 0.60 recall), widen
    # left so the shorter curve is still visible for comparison.
    left, right = _XLIM
    if plotted_max_recall is not None and plotted_max_recall < left:
        # Every curve is below the zoom window; fall back to the plotted range.
        assert plotted_min_recall is not None
        left = max(0.0, plotted_min_recall - 0.02)
        right = min(1.002, plotted_max_recall + 0.02)
    elif plotted_min_recall is not None and plotted_min_recall < left:
        # At least one stage is short; drop the left edge to that stage's max
        # recall so its whole Pareto is visible.
        stage_max_below_zoom = None
        for stage in stages:
            fx, _ = _stage_pareto_xy(df[df["variant"] == stage])
            if fx.size == 0:
                continue
            mx = float(np.nanmax(fx))
            if mx < _XLIM[0]:
                stage_max_below_zoom = mx if stage_max_below_zoom is None else max(stage_max_below_zoom, mx)
        if stage_max_below_zoom is not None:
            left = max(0.0, min(_XLIM[0], stage_max_below_zoom - 0.02))
    ax.set_xlim(left, right)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_title:
        ax.set_title(dataset, fontsize=15)
    if show_legend:
        ps._legend(ax)
    return stages


def _plot_pareto(df: pd.DataFrame, dataset: str, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    stages = _plot_pareto_panel(ax, df, dataset, show_ylabel=True, show_legend=True)
    if not stages:
        print(f"  [skip pareto for {dataset}: no known stage variants]")
        plt.close(fig)
        return

    fig.suptitle(
        f"{dataset}: MV-IVF Optimizations (Recall vs Latency)",
        fontsize=17,
        y=0.97,
    )
    fig.subplots_adjust(top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_pareto_combined(
    dfs: dict[str, pd.DataFrame], datasets: list[str], out_path: pathlib.Path,
) -> None:
    """Side-by-side Pareto panels (one column per dataset) for the paper.

    Same two-row-gridspec "dedicated legend band" convention as
    ``_plot_ladder_combined`` below — a shared stage legend on top, then one
    panel per dataset underneath, each keeping its own x-limits / y-scale
    (datasets differ enough in latency range that forcing a shared y-axis
    would flatten the shorter curves). Color scheme (viridis per stage) is
    untouched — only the layout is new.
    """
    plot_datasets = [d for d in datasets if d in dfs]
    if len(plot_datasets) < 2:
        print(f"  [skip combined pareto: need >=2 loaded datasets, got {len(plot_datasets)}]")
        return

    n = len(plot_datasets)
    fig = plt.figure(figsize=(6.6 * n, 5.25))
    gs = fig.add_gridspec(
        2, n,
        height_ratios=[0.17, 1.0],
        hspace=0.14,
        wspace=0.12,
        left=0.08, right=0.98, bottom=0.12, top=0.99,
    )

    axes = []
    plotted_any = False
    for i, dataset in enumerate(plot_datasets):
        ax = fig.add_subplot(gs[1, i])
        stages = _plot_pareto_panel(
            ax, dfs[dataset], dataset,
            show_ylabel=(i == 0), show_legend=False, show_title=True,
        )
        plotted_any = plotted_any or bool(stages)
        axes.append(ax)

    if not plotted_any:
        print("  [skip combined pareto: no dataset had known stage variants]")
        plt.close(fig)
        return

    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis("off")
    handles, labels = ps._dedup_legend_handles(axes)
    wrapped_labels = [_wrap_legend_label(lbl) for lbl in labels]
    leg_kwargs = {**_legend_kwargs(len(labels)), "fontsize": 14, "handlelength": 2.2, "markerscale": 1.3}
    leg_ax.legend(
        handles,
        wrapped_labels,
        loc="center",
        **leg_kwargs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  wrote {out_path} ({n} panel(s): {', '.join(plot_datasets)})")


def _compute_latency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stages = [s for s in _STAGE_ORDER if s in df["variant"].unique()]
    for stage in stages:
        sub = df[df["variant"] == stage]
        fx, fy = _stage_pareto_xy(sub)
        for r in _RECALL_THRESHOLDS:
            lat = _latency_at_recall(fx, fy, r) if fx.size else None
            rows.append({
                "stage": stage,
                "threshold": r,
                "latency_ms": np.nan if lat is None else lat,
            })
    return pd.DataFrame(rows)


def _recall_file_tag(recall: float) -> str:
    return f"r{int(round(recall * 100))}"


def _recall_complete(pivot: pd.DataFrame, stages: list[str], recall: float) -> bool:
    if recall not in pivot.columns:
        return False
    col = pivot.loc[stages, recall]
    return bool(col.notna().all() and (col > 0).all())


def _format_latency_ms(lat: float) -> str:
    return f"{lat:.0f}" if lat >= 100 else f"{lat:.1f}"


def _wrap_legend_label(label: str, max_chars: int = 16) -> str:
    """Break long legend labels; continuation lines hang-indent after ``+ ``."""
    hang = "   "  # hang-indent after "+ " (3 spaces; proportional font needs extra)

    if label.startswith("+ "):
        rest = label[2:]
        paren = re.match(r"^(.+?)\s+(\([^)]+\))$", rest)
        if paren:
            first = f"+ {paren.group(1).strip()}"
            return f"{first}\n{hang}{paren.group(2)}"

        words = rest.split()
        lines: list[str] = []
        cur = ""
        limit = max_chars - 2
        for word in words:
            chunk = f"{cur} {word}".strip()
            if len(chunk) <= limit:
                cur = chunk
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if not lines:
            return label
        out = [f"+ {lines[0]}"]
        out.extend(f"{hang}{ln}" for ln in lines[1:])
        return "\n".join(out)

    if len(label) <= max_chars:
        return label
    words = label.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        chunk = f"{cur} {word}".strip()
        if len(chunk) <= max_chars:
            cur = chunk
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _ladder_legend_handles(stages: list[str], colors: list) -> list:
    return [
        mpatches.Patch(
            facecolor=colors[i],
            edgecolor="white",
            label=_wrap_legend_label(_STAGE_LABELS[stages[i]]),
        )
        for i in range(len(stages))
    ]


def _legend_kwargs(ncol: int) -> dict:
    return dict(
        ncol=ncol,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
        columnspacing=1.0,
        handletextpad=0.35,
        labelspacing=0.25,
    )


def _add_ladder_fig_legend(
    fig,
    stages: list[str],
    colors: list,
    *,
    anchor_y: float = 0.93,
) -> None:
    """One-row figure legend above a single axes (bbox anchor)."""
    handles = _ladder_legend_handles(stages, colors)
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        **_legend_kwargs(len(stages)),
    )


def _plot_ladder_combined(
    fig,
    stages: list[str],
    colors: list,
    complete: list[float],
    pivot: pd.DataFrame,
    dataset: str,
    k_val: int,
) -> None:
    """Two-row layout: dedicated legend band, then recall panels (avoids tight bbox fights)."""
    ncol = len(complete)
    gs = fig.add_gridspec(
        2,
        ncol,
        height_ratios=[0.30, 1.0],
        hspace=0.13,
        left=0.10,
        right=0.98,
        bottom=0.1,
        top=0.94,
    )
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis("off")
    handles = _ladder_legend_handles(stages, colors)
    leg_ax.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center",
        **_legend_kwargs(len(stages)),
    )

    for j, recall in enumerate(complete):
        ax = fig.add_subplot(gs[1, j])
        lats = pivot[recall].to_numpy(dtype=float)
        _plot_ladder_panel(
            ax, stages, lats, colors, recall,
            show_ylabel=(j == 0),
            show_panel_title=True,
        )

    fig.suptitle(_ladder_title(dataset, k_val), fontsize=17, y=0.96)


def _plot_ladder_panel(
    ax,
    stages: list[str],
    lats: np.ndarray,
    colors: list,
    recall: float,
    *,
    show_ylabel: bool,
    show_panel_title: bool,
) -> None:
    """Vertical ladder: touching bars, latency (ms) on tops."""
    n = len(stages)
    xs = _BAR_SIDE_PAD + np.arange(n, dtype=float)
    heights = np.where(np.isfinite(lats), lats, 0.0)

    ax.bar(
        xs,
        heights,
        width=_BAR_WIDTH,
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    finite = [float(v) for v in lats if np.isfinite(v) and v > 0]
    ymax = max(finite) if finite else 1.0
    ymax *= 1.14

    for si in range(n):
        lat = float(lats[si])
        if not (np.isfinite(lat) and lat > 0):
            continue
        ax.text(
            xs[si] + _BAR_WIDTH / 2.0,
            lat + ymax * 0.02,
            _format_latency_ms(lat),
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.15",
            zorder=5,
        )

    ax.set_xticks(xs + _BAR_WIDTH / 2.0)
    ax.set_xticklabels([])
    ax.set_xlim(0, _BAR_SIDE_PAD + n + 0.35)
    ax.set_ylim(0, ymax)
    if show_ylabel:
        ax.set_ylabel(ps._stage_ylabel("latency", latency_ms=True))
    if show_panel_title:
        ax.set_title(_ladder_panel_title(recall), fontsize=14, pad=6)
    ax.grid(True, axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_waterfall_ladders(
    df: pd.DataFrame, dataset: str, out_dir: pathlib.Path,
) -> pd.DataFrame:
    """Per-recall PDFs + combined row of complete regimes only."""
    lat_tab = _compute_latency_table(df)
    if lat_tab.empty or lat_tab["latency_ms"].isna().all():
        print(f"  [skip ladders for {dataset}: no latency-at-recall data]")
        return lat_tab

    stages = [s for s in _STAGE_ORDER if s in lat_tab["stage"].unique()]
    if not stages:
        return lat_tab

    pivot = lat_tab.pivot(index="stage", columns="threshold", values="latency_ms")
    pivot = pivot.reindex(stages)
    n_s = len(stages)
    colors = [_stage_color(i, n_s) for i in range(n_s)]
    k_val = int(df["k"].dropna().iloc[0]) if "k" in df.columns and not df["k"].dropna().empty else 10

    complete: list[float] = []

    for recall in _RECALL_THRESHOLDS:
        if recall not in pivot.columns:
            continue
        lats = pivot[recall].to_numpy(dtype=float)
        if not np.any(np.isfinite(lats) & (lats > 0)):
            continue

        tag = _recall_file_tag(recall)
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        _plot_ladder_panel(
            ax, stages, lats, colors, recall,
            show_ylabel=True,
            show_panel_title=False,
        )
        fig.suptitle(_ladder_title(dataset, k_val, recall), fontsize=17, y=0.93)
        _add_ladder_fig_legend(fig, stages, colors, anchor_y=0.855)
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.74)
        out_path = out_dir / f"{dataset}_ladder_{tag}.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"  wrote {out_path}")

        if _recall_complete(pivot, stages, recall):
            complete.append(recall)

    if complete:
        ncol = len(complete)
        fig = plt.figure(figsize=(5.6 * ncol, 5.4))
        _plot_ladder_combined(
            fig, stages, colors, complete, pivot, dataset, k_val,
        )
        out_path = out_dir / f"{dataset}_ladder_combined.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"  wrote {out_path} ({ncol} panel(s): {', '.join(_recall_file_tag(r) for r in complete)})")
    else:
        print(f"  [skip combined ladder for {dataset}: no recall regime with all stages]")

    return lat_tab


def _write_table(lat_tab: pd.DataFrame, dataset: str, out_path: pathlib.Path) -> None:
    if lat_tab.empty:
        return
    stages = [s for s in _STAGE_ORDER if s in lat_tab["stage"].unique()]
    pivot = lat_tab.pivot(index="stage", columns="threshold", values="latency_ms").reindex(stages)
    baseline = pivot.iloc[0]

    lines = [f"# Optimizations ladder — {dataset}", ""]
    header = ["Stage"]
    for r in _RECALL_THRESHOLDS:
        header += [f"lat(ms) @ {r:.2f}", "step ×", "vs baseline ×"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    prev = baseline.copy()
    for stage in stages:
        cells = [_STAGE_LABELS.get(stage, stage)]
        for r in _RECALL_THRESHOLDS:
            lat = pivot.loc[stage, r] if r in pivot.columns else np.nan
            base_lat = baseline[r] if r in baseline.index else np.nan
            prev_lat = prev[r] if r in prev.index else np.nan
            lat_str = f"{lat:.2f}" if np.isfinite(lat) else "—"
            step_str = (
                f"{prev_lat / lat:.2f}"
                if (np.isfinite(lat) and np.isfinite(prev_lat) and lat > 0)
                else "—"
            )
            cum_str = (
                f"{base_lat / lat:.2f}"
                if (np.isfinite(lat) and np.isfinite(base_lat) and lat > 0)
                else "—"
            )
            cells += [lat_str, step_str, cum_str]
        lines.append("| " + " | ".join(cells) + " |")
        prev = pivot.loc[stage].copy()

    lines += [
        "",
        "*Step ×* = previous-stage latency / this-stage latency ( >1 means faster).",
        "*vs baseline ×* = baseline latency / this-stage latency.",
        "Cells with `—` mean the stage did not reach that recall in the sweep.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def _list_datasets(results_root: pathlib.Path) -> list[str]:
    if not results_root.exists():
        return []
    return sorted([
        p.name for p in results_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
    ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=None)
    p.add_argument(
        "--results",
        type=pathlib.Path,
        default=REPO_ROOT / "experiments" / "optimizations" / "results",
    )
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else args.results / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets
        else _list_datasets(args.results)
    )
    if not datasets:
        print(f"[warn] no datasets under {args.results}; nothing to plot.")
        return 0

    loaded: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        try:
            df = _load_dataset(args.results, ds)
        except SystemExit as e:
            print(f"[skip {ds}] {e}")
            continue
        print(f"[{ds}] {len(df)} rows, {df['variant'].nunique()} stages")
        loaded[ds] = df
        _plot_pareto(df, ds, out_dir / f"{ds}_pareto.pdf")
        lat_tab = _plot_waterfall_ladders(df, ds, out_dir)
        _write_table(lat_tab, ds, out_dir / f"{ds}_speedup_table.md")

    if len(loaded) >= 2:
        combined_name = "_".join(loaded.keys()) + "_paper_pareto.pdf"
        _plot_pareto_combined(loaded, list(loaded.keys()), out_dir / combined_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
