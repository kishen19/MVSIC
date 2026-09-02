#!/usr/bin/env python3
"""Produce stage-wise recall plots from `latency_*.csv` files.

Usage:
    scripts/plot.py \
        --results experiments/mvivf_ablation/results/nq500k/mvivf/ \
        --group-by k_per_level \
        --out-dir experiments/mvivf_ablation/results/nq500k/mvivf/

Expected CSV columns:
    k, nprobes, num_rerank, recall_1_k, recall_k_k, QPS_seq/QPS_par, avg_cmps
The loader also accepts lowercase qps_seq/qps_par and normalizes them.
Latency(ms) = 1000 / QPS_seq.

Outputs:
    <out-dir>/<prefix>_pareto.pdf    (x=recall_k_k, y=QPS_seq Pareto by group)
    <out-dir>/<prefix>_latency.pdf   (x=recall_k_k, y=latency_ms)
    <out-dir>/<prefix>_cmps.pdf      (x=recall_k_k, y=avg_cmps)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

try:
    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib to use this script")

# Share styling helpers with the main latency / batch plots so ablations look
# identical (fonts, legend box, spine visibility, Pareto x-axis limits, etc.).
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
_BUILD_SCRIPTS = REPO_ROOT / "experiments" / "builds" / "scripts"
if str(_BUILD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BUILD_SCRIPTS))

import plot_stage as ps  # noqa: E402

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

_FAMILY_LABEL = {
    "mvivf":        "MV-IVF",
    "mvivf_spill":  "MV-IVF-Spill",
    "mvivf_flat":   "MV-IVF-Flat",
}

# Pretty math symbols for ablation knobs, used in titles + legend entries.
# ``None`` means "render as plain text, no $...$ wrapping" (e.g. niters).
_PARAM_SYMBOL: dict[str, str | None] = {
    "k_per_level": r"b",
    "max_leaf_size": r"\ell_{\max}",
    "niters": None,
}


def _param_display(group_col: str) -> str:
    """Pretty title fragment for a group column, e.g. ``$b$`` or ``niters``."""
    if group_col not in _PARAM_SYMBOL:
        return group_col
    sym = _PARAM_SYMBOL[group_col]
    return group_col if sym is None else f"${sym}$"


def _param_value_label(group_col: str, value) -> str:
    """Pretty legend entry for one swept value, e.g. ``$b=10$`` or ``niters=5``."""
    sym = _PARAM_SYMBOL.get(group_col)
    if sym is None:
        return f"{group_col}={value}"
    return f"${sym}={value}$"


# Cool -> warm along the sorted sweep values (same convention as
# experiments/optimizations/scripts/plot.py's ``_stage_color``), so e.g.
# k_per_level=0 is always the darkest curve and the largest value is always
# the brightest, consistently across every ablation plot.
def _ordered_color(i: int, n: int) -> tuple:
    cmap = plt.colormaps.get_cmap("viridis")
    if n <= 1:
        return cmap(0.5)
    t = 0.12 + 0.76 * (i / (n - 1))
    return cmap(t)


# Fixed legend corner per y-metric (instead of matplotlib's per-axes "best"),
# so e.g. the fiqa and hotpotqa panels for the *same* stage/param put the
# legend in the same place -- "best" independently picks whichever corner is
# locally emptiest and disagrees across datasets even for the same knob.
# Chosen from the metrics' monotonic shape: latency/cmps rise with recall
# (empty top-left), QPS Pareto falls with recall (empty top-right).
_LEGEND_LOC_BY_YCOL = {
    "latency_ms": "upper left",
    "avg_cmps": "upper left",
    "QPS_seq": "upper right",
}


def _legend(ax, loc: str = "best") -> None:
    ax.legend(
        loc=loc,
        fontsize=15,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
        borderpad=0.3,
        handlelength=1.6,
        handletextpad=0.35,
        labelspacing=0.2,
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "qps_seq" in df.columns and "QPS_seq" not in df.columns:
        rename["qps_seq"] = "QPS_seq"
    if "qps_par" in df.columns and "QPS_par" not in df.columns:
        rename["qps_par"] = "QPS_par"
    if rename:
        df = df.rename(columns=rename)
    return df


# Stage knobs that can appear in canonical build_config names.  Keys are the
# long names used as the `--group-by` argument; values are (regex, capture
# index) matched against `build_config` (or fallback path strings); the
# capture group holds the integer value.
#
# Build-config naming convention (mvivf):
#     mvivf_k<K>_l<L>_d<D>_n<N>_s<S>_mpcc<M>_mpcik<MK>
# Only the suffixes relevant to the current stage need to be present; missing
# suffixes leave the corresponding derived column NaN.
#
# For mvivf_spill use `_spill_<N>_<N2>` -> num_spill / num_spill_l2, or `_s<N>`
# for the generic `s` knob (same sweep axis as older configs).
_KNOB_PATTERNS: dict[str, tuple[str, int]] = {
    "k_per_level": (r"_k(\d+)(?:_|$)", 0),
    "max_leaf_size": (r"_l(\d+)(?:_|$)", 0),
    "max_depth": (r"_d(\d+)(?:_|$)", 0),
    "niters": (r"_nit(\d+)(?:_|$)", 0),
    "s": (r"_s(\d+)(?:_|$)", 0),
    "max_point_clouds_per_cluster": (r"_mpcc(\d+)(?:_|$)", 0),
    "max_points_per_centroid_inner_kmeans": (r"_mpcik(\d+)(?:_|$)", 0),
    # mvivf_spill: first number after _spill_, optional second (e.g. ..._spill_10_1_...).
    "num_spill": (r"_spill_(\d+)(?:_|$)", 0),
    "num_spill_l2": (r"_spill_(\d+)_(\d+)(?:_|$)", 1),
}


def _derive_knob_column(
    series: pd.Series, pattern: str, capture_group: int = 0
) -> pd.Series:
    extracted = series.astype(str).str.extract(pattern)[capture_group]
    return extracted.apply(
        lambda x: int(x) if isinstance(x, str) and x.isdigit() else x
    )


def _populate_knobs(df: pd.DataFrame, source_col: str = "build_config") -> pd.DataFrame:
    if source_col not in df.columns:
        return df
    for col, (pat, group_i) in _KNOB_PATTERNS.items():
        if col in df.columns:
            continue
        df[col] = _derive_knob_column(df[source_col], pat, group_i)
    return df


def _annotate_from_path(base: pathlib.Path, csv_file: pathlib.Path) -> dict:
    """
    Infer metadata from the benchmark_search.py output tree:
      <results>/<index_name>/<build_name>/<variant_name?>/<search_name>/latency_*.csv

    variant_name is optional (raw variant).
    """
    rel = csv_file.relative_to(base)
    parts = rel.parts
    meta = {}
    # Supported relative layouts (base can be anywhere above these):
    #   <build>/<search>/<csv>
    #   <build>/<variant>/<search>/<csv>
    #   <index>/<build>/<search>/<csv>
    #   <index>/<build>/<variant>/<search>/<csv>
    if len(parts) >= 3:
        meta["search_name"] = parts[-2]
        # Try to find the build folder as the nearest ancestor that looks
        # like mvivf_kXX_...; fallback to a position-based heuristic.
        build_idx = None
        for i in range(len(parts) - 3, -1, -1):
            if re.search(r"_k\d+(?:_|$)", parts[i]):
                build_idx = i
                break
        if build_idx is None:
            # csv at -1, search at -2, so build is normally -3
            build_idx = max(0, len(parts) - 3)
        meta["build_config"] = parts[build_idx]
        if build_idx > 0:
            meta["index_name"] = parts[build_idx - 1]
        # Variant exists when there is exactly one path component between
        # build and search.
        if build_idx + 2 == len(parts) - 1:
            meta["variant_name"] = parts[build_idx + 1]
        else:
            meta["variant_name"] = "raw"
    return meta


def load_csvs(path: pathlib.Path) -> pd.DataFrame:
    # Support both:
    #   - old flat layout: <dir>/latency_*.csv
    #   - current nested layout under results/<dataset>/<method>/...
    files = sorted(path.rglob("latency_*.csv"))
    if not files:
        raise SystemExit(f"no latency_*.csv found recursively under {path}")
    chunks = []
    for f in files:
        df = pd.read_csv(f)
        df = _normalize_columns(df)
        meta = _annotate_from_path(path, f)
        df["source_relpath"] = str(f.relative_to(path))
        df["source_abspath"] = str(f)
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        # Fallback: if build_config is still missing, recover from full path.
        if "build_config" not in df.columns:
            m = re.search(r"([^/]+_k\d+[^/]*)", str(f))
            if m:
                df["build_config"] = m.group(1)
        chunks.append(df)
    out = pd.concat(chunks, ignore_index=True)
    # Derive every known stage knob column from canonical build names such as
    # mvivf_k0_l500_d3_n5_s2_mpcc100_mpcik20.  Only suffixes present in the
    # name will yield non-NaN values; missing suffixes are NaN and the user
    # must group by a knob that *is* swept in this stage's configs.
    out = _populate_knobs(out, "build_config")
    return out


def _ensure_group_col(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df = df.copy()
    have_col = group_col in df.columns and df[group_col].notna().any()
    if not have_col:
        # Try to derive from build_config / source paths using the canonical
        # short-name convention (see _KNOB_PATTERNS above).
        entry = _KNOB_PATTERNS.get(group_col)
        if entry is not None:
            pattern, group_i = entry
            for src_col in ("build_config", "source_relpath", "source_abspath"):
                if src_col not in df.columns:
                    continue
                derived = _derive_knob_column(df[src_col], pattern, group_i)
                if derived.notna().any():
                    df[group_col] = derived
                    break

    # Stage-5 trick: plain mvivf builds (no `_spill_<n>`) act as the
    # `num_spill = 1` baseline against mvivf_spill rows. Backfill those rows
    # so they show up as a single curve next to the spill-ratio sweep.
    if group_col == "num_spill" and "build_config" in df.columns:
        if group_col not in df.columns:
            df[group_col] = pd.NA
        is_plain = (
            df["build_config"].astype(str).str.startswith("mvivf_")
            & ~df["build_config"].astype(str).str.contains("_spill_")
        )
        df.loc[is_plain & df[group_col].isna(), group_col] = 1

    # Stage-6 default: when grouping by `query_compression` and the column was
    # not written (older CSVs), backfill "None" so they land in the no-QC
    # baseline curve.
    if group_col == "query_compression":
        if group_col not in df.columns:
            df[group_col] = "None"
        df[group_col] = df[group_col].fillna("None").replace("", "None")

    if group_col not in df.columns or not df[group_col].notna().any():
        raise SystemExit(
            f"group column '{group_col}' not found and could not be derived "
            f"from build_config. Available columns: {sorted(df.columns)}"
        )
    return df


def _series_keys(df: pd.DataFrame, group_col: str,
                 extra_label_col: str | None) -> list[str]:
    """Group columns to split curves by. Always includes ``group_col``; when
    ``extra_label_col`` is set and present in the frame it's prepended so that
    e.g. (index_name, query_compression) becomes one curve each."""
    keys = []
    if extra_label_col and extra_label_col in df.columns:
        keys.append(extra_label_col)
    keys.append(group_col)
    return keys


def _series_label(key, group_col: str, extra_label_col: str | None) -> str:
    # pandas' ``groupby(<list-of-columns>)`` always returns a tuple matching
    # the length of that list -- even for a single-column list -- so a plain
    # scalar ``k_per_level=0`` group key arrives here as ``(0,)``. Normalize
    # first so single-knob ablations never render as e.g. ``k_per_level=(0,)``.
    if not isinstance(key, tuple):
        key = (key,)
    if extra_label_col is None:
        (group_val,) = key
        return _param_value_label(group_col, group_val)
    extra_val, group_val = key
    return f"{extra_val} / {_param_value_label(group_col, group_val)}"


# Recall window every ablation plot zooms into. Anything outside this band
# is dropped before plotting (a recall column is assumed for the x-axis).
_RECALL_XLIM = (0.6, 1.0)
# Tiny visual pad so a marker at recall == 1 is not clipped by the spine.
_RECALL_XLIM_PAD_RIGHT = 1.002


def _filter_to_recall_window(
    df: pd.DataFrame, x_col: str, lo: float, hi: float
) -> pd.DataFrame:
    if x_col not in df.columns:
        return df
    s = pd.to_numeric(df[x_col], errors="coerce")
    return df[s.between(lo, hi)]


def _recall_k_k_label(df: pd.DataFrame) -> str:
    if "k" in df.columns:
        vals = sorted({int(v) for v in df["k"].dropna().astype(int).unique()})
        if len(vals) == 1:
            return f"Recall-${vals[0]}$@${vals[0]}$"
    return r"Recall-$k$@$k$"


def _min_envelope(xs: pd.Series, ys: pd.Series,
                  min_recall_spacing: float = 0.005) -> tuple[pd.Series, pd.Series]:
    """Lower envelope of (recall, cost) points: for each recall, keep min y.

    Symmetric to :func:`_pareto_curve` but for cost metrics (latency, bytes
    accessed) where lower is better. Guarantees a monotone curve so the
    plotted line doesn't zigzag through overlapping sweeps (e.g. different
    ``num_rerank`` values at the same nprobes)."""
    if xs.empty:
        return xs, ys
    tmp = pd.DataFrame({"x": xs.astype(float), "y": ys.astype(float)}).dropna()
    if tmp.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    tmp = tmp.loc[tmp.groupby("x")["y"].idxmin()]

    tmp = tmp.sort_values(by=["x", "y"], ascending=[False, True])
    front_rows = []
    best_y = float("inf")
    for _, r in tmp.iterrows():
        if float(r["y"]) < best_y:
            front_rows.append(r)
            best_y = float(r["y"])
    if not front_rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    front = pd.DataFrame(front_rows).sort_values("x")

    if min_recall_spacing > 0 and len(front) > 2:
        simplified = [front.iloc[0]]
        for i in range(1, len(front) - 1):
            if abs(float(front.iloc[i]["x"]) - float(simplified[-1]["x"])) > min_recall_spacing:
                simplified.append(front.iloc[i])
        simplified.append(front.iloc[-1])
        front = pd.DataFrame(simplified).drop_duplicates(subset=["x", "y"], keep="first")
    return front["x"], front["y"]


def _plot_xy_panel(
    ax,
    df: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    *,
    extra_label_col: str | None = None,
    show_scatter: bool = False,
) -> int:
    """Draw the per-swept-value min-envelope curves for one (dataset, stage)
    onto ``ax``. Shared by the standalone ``_plot_xy`` and the grid plots so
    both stay pixel-identical for the curves they have in common. Returns
    the number of series actually plotted.
    """
    keys = _series_keys(df, group_col, extra_label_col)
    df = _filter_to_recall_window(df, x_col, *_RECALL_XLIM)
    groups = list(df.groupby(keys))
    n = len(groups)
    plotted = 0
    for i, (key, sub) in enumerate(groups):
        sub = sub.dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        color = _ordered_color(i, n)
        if show_scatter:
            ax.scatter(sub[x_col], sub[y_col], s=20, alpha=0.25, zorder=1, color=color)
        fx, fy = _min_envelope(sub[x_col], sub[y_col])
        if len(fx) == 0:
            continue
        ax.plot(fx, fy, marker="o", markersize=5.4, linewidth=2.8, zorder=3,
                color=color, label=_series_label(key, group_col, extra_label_col))
        plotted += 1
    ax.set_xlim(_RECALL_XLIM[0], _RECALL_XLIM_PAD_RIGHT)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return plotted


def _plot_xy(df: pd.DataFrame, group_col: str, x_col: str, y_col: str,
             x_label: str, y_label: str, title: str, out_path: pathlib.Path,
             extra_label_col: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_xy_panel(ax, df, group_col, x_col, y_col, extra_label_col=extra_label_col)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _legend(ax, loc=_LEGEND_LOC_BY_YCOL.get(y_col, "best"))
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")
    plt.close(fig)


def _pareto_curve(xs: pd.Series, ys: pd.Series, min_recall_spacing: float = 0.005) -> tuple[pd.Series, pd.Series]:
    """Build a simplified Pareto curve (maximize recall and QPS)."""
    if xs.empty:
        return xs, ys

    # Keep max QPS for each recall value.
    tmp = pd.DataFrame({"x": xs.astype(float), "y": ys.astype(float)})
    tmp = tmp.loc[tmp.groupby("x")["y"].idxmax()]

    # True Pareto front.
    tmp = tmp.sort_values(by=["x", "y"], ascending=[False, False])
    front_rows = []
    best_y = float("-inf")
    for _, r in tmp.iterrows():
        if float(r["y"]) > best_y:
            front_rows.append(r)
            best_y = float(r["y"])
    if not front_rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    front = pd.DataFrame(front_rows).sort_values("x")

    if min_recall_spacing > 0 and len(front) > 2:
        simplified = [front.iloc[0]]
        for i in range(1, len(front) - 1):
            if abs(float(front.iloc[i]["x"]) - float(simplified[-1]["x"])) > min_recall_spacing:
                simplified.append(front.iloc[i])
        simplified.append(front.iloc[-1])
        front = pd.DataFrame(simplified).drop_duplicates(subset=["x", "y"], keep="first")
    return front["x"], front["y"]


def _plot_qps_pareto(df: pd.DataFrame, group_col: str, out_path: pathlib.Path,
                     title: str, extra_label_col: str | None = None) -> None:
    if "QPS_seq" not in df.columns:
        raise SystemExit("Missing QPS_seq column in merged CSVs.")
    if "recall_k_k" not in df.columns:
        raise SystemExit("Missing recall_k_k column in merged CSVs.")

    fig, ax = plt.subplots(figsize=(7, 5))
    keys = _series_keys(df, group_col, extra_label_col)
    df = _filter_to_recall_window(df, "recall_k_k", *_RECALL_XLIM)
    groups = list(df.groupby(keys))
    n = len(groups)
    for i, (key, sub) in enumerate(groups):
        sub = sub.dropna(subset=["recall_k_k", "QPS_seq"])
        if sub.empty:
            continue
        color = _ordered_color(i, n)
        fx, fy = _pareto_curve(sub["recall_k_k"], sub["QPS_seq"])
        if len(fx) == 0:
            continue
        ax.plot(fx, fy, marker="o", markersize=5.4, linewidth=2.8, zorder=3,
                color=color, label=_series_label(key, group_col, extra_label_col))
    ax.set_xlim(_RECALL_XLIM[0], _RECALL_XLIM_PAD_RIGHT)

    ax.set_xlabel(_recall_k_k_label(df))
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _legend(ax, loc=_LEGEND_LOC_BY_YCOL.get("QPS_seq", "best"))
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")
    plt.close(fig)


def _ablation_title(dataset: str | None, method_label: str, group_col: str,
                    extra_label_col: str | None) -> str:
    """Uniform ablation title matching latency-plot style.

    e.g. ``hotpotqa: MV-IVF Ablation (niters)`` or
    ``hotpotqa: MV-IVF Ablation (index_name x query_compression)``.
    """
    param = (
        _param_display(group_col)
        if extra_label_col is None
        else f"{extra_label_col} x {_param_display(group_col)}"
    )
    head = f"{method_label} Ablation ({param})"
    return f"{dataset}: {head}" if dataset else head


def plot(df: pd.DataFrame, group_col: str, out_dir: pathlib.Path,
         prefix: str = "stage1", extra_label_col: str | None = None,
         *, dataset: str | None = None,
         method_label: str = "MV-IVF") -> None:
    if "QPS_seq" in df.columns:
        df = df.assign(latency_ms=1000.0 / df["QPS_seq"])
    else:
        raise SystemExit("Missing QPS_seq column in merged CSVs.")
    if "avg_cmps" not in df.columns:
        raise SystemExit("Missing avg_cmps column in merged CSVs.")
    df = _ensure_group_col(df, group_col)

    # For stage selection we use Recall-k@k.
    recall_x = "recall_k_k"
    if recall_x not in df.columns:
        raise SystemExit("Missing recall_k_k column in merged CSVs.")

    title = _ablation_title(dataset, method_label, group_col, extra_label_col)
    _plot_qps_pareto(
        df,
        group_col=group_col,
        out_path=out_dir / f"{prefix}_pareto.pdf",
        title=title,
        extra_label_col=extra_label_col,
    )
    _plot_xy(
        df,
        group_col=group_col,
        x_col=recall_x,
        y_col="latency_ms",
        x_label=_recall_k_k_label(df),
        y_label="Latency (ms)",
        title=title,
        out_path=out_dir / f"{prefix}_latency.pdf",
        extra_label_col=extra_label_col,
    )
    _plot_xy(
        df,
        group_col=group_col,
        x_col=recall_x,
        y_col="avg_cmps",
        x_label=_recall_k_k_label(df),
        y_label="Bytes accessed per query",
        title=title,
        out_path=out_dir / f"{prefix}_cmps.pdf",
        extra_label_col=extra_label_col,
    )


# Stage -> swept knob, for the grid plots (matches
# scripts/run_ablation.sh:default_group_by_for_stage).
_STAGE_GROUP_COL = {
    "stage1": "k_per_level",
    "stage2": "max_leaf_size",
    "stage4": "niters",
}


def _load_stage_df(
    results_root: pathlib.Path, dataset: str, stage: str, method: str = "mvivf",
) -> pd.DataFrame:
    base = results_root / dataset / stage / method
    df = load_csvs(base)
    df = df.assign(latency_ms=1000.0 / df["QPS_seq"])
    df = _ensure_group_col(df, _STAGE_GROUP_COL[stage])
    return df


def _plot_ablation_grid(
    results_root: pathlib.Path,
    datasets: list[str],
    stages: list[str],
    out_path: pathlib.Path,
    *,
    transpose: bool = False,
    method: str = "mvivf",
) -> None:
    """Grid of latency-vs-recall ablation panels, one column per stage-knob
    sweep and one row per dataset (``transpose=True`` swaps the two).

    Colors/legend/curve style exactly match the standalone ``*_latency.pdf``
    plots (both call ``_plot_xy_panel``). Column headers carry the "Ablating
    {param}" title, row headers (rotated) carry whichever axis isn't the
    column -- dataset name or the other stage's param -- so both orientations
    stay self-describing without repeating the same label six times.
    """
    dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for ds in datasets:
        for stage in stages:
            try:
                dfs[(ds, stage)] = _load_stage_df(results_root, ds, stage, method)
            except SystemExit as e:
                print(f"  [skip grid cell {ds}/{stage}: {e}]")

    if not dfs:
        print(f"  [skip {out_path.name}: no ablation data loaded]")
        return

    row_is_dataset = not transpose
    row_keys, col_keys = (datasets, stages) if row_is_dataset else (stages, datasets)
    n_rows, n_cols = len(row_keys), len(col_keys)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.7 * n_rows),
        squeeze=False,
    )

    for r, row_key in enumerate(row_keys):
        for c, col_key in enumerate(col_keys):
            ds, stage = (row_key, col_key) if row_is_dataset else (col_key, row_key)
            ax = axes[r][c]
            df = dfs.get((ds, stage))
            if df is None:
                ax.axis("off")
                continue
            group_col = _STAGE_GROUP_COL[stage]
            _plot_xy_panel(ax, df, group_col, "recall_k_k", "latency_ms")
            if r == n_rows - 1:
                ax.set_xlabel(_recall_k_k_label(df))
            if c == 0:
                ax.set_ylabel("Latency (ms)")
            if r == 0:
                col_title = f"Ablating {_param_display(group_col)}" if row_is_dataset else ds
                ax.set_title(col_title, fontsize=15)
            _legend(ax, loc=_LEGEND_LOC_BY_YCOL["latency_ms"])

    fig.tight_layout(rect=(0.018, 0.0, 1.0, 1.0))

    for r, row_key in enumerate(row_keys):
        label = row_key if row_is_dataset else f"Ablating {_param_display(_STAGE_GROUP_COL[row_key])}"
        pos = axes[r][0].get_position()
        fig.text(
            0.004, (pos.y0 + pos.y1) / 2, label,
            rotation=90, va="center", ha="center", fontsize=15,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  wrote {out_path} ({n_rows}x{n_cols}: rows={row_keys}, cols={col_keys})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid", action="store_true",
        help=("Generate the multi-dataset x multi-stage latency ablation "
              "grid(s) instead of the single --results/--group-by plot. "
              "Writes both the default orientation and its transpose."),
    )
    p.add_argument("--grid-datasets", default="fiqa,hotpotqa")
    p.add_argument("--grid-stages", default="stage1,stage2,stage4")
    p.add_argument(
        "--grid-results", type=pathlib.Path, default=None,
        help="Ablation results root (default: experiments/mvivf_ablation/results).",
    )
    p.add_argument(
        "--grid-out-dir", type=pathlib.Path, default=None,
        help="Output dir for the grid PDFs (default: <grid-results>/_plots).",
    )
    p.add_argument("--results", type=pathlib.Path, default=None)
    p.add_argument("--group-by", default="build_config",
                   help="CSV column to group curves by.")
    p.add_argument(
        "--label-by",
        default=None,
        help=("Optional secondary column whose value is prefixed onto each "
              "curve label (e.g. `index_name` for stage 5/6/7 plots that mix "
              "mvivf and mvivf_spill in the same axes)."),
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help=("Output directory for stage1_latency.pdf and stage1_cmps.pdf "
              "(default: --results)."),
    )
    p.add_argument(
        "--prefix",
        default="stage1",
        help="Output filename prefix (default: stage1).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset name to include in plot titles (e.g. 'hotpotqa').",
    )
    p.add_argument(
        "--method-label",
        default="MV-IVF",
        help="Method label for plot titles (default: MV-IVF; e.g. MV-IVF-Spill).",
    )
    args = p.parse_args()

    if args.grid:
        grid_results = (
            args.grid_results if args.grid_results is not None
            else HERE.parent / "results"
        )
        grid_out_dir = (
            args.grid_out_dir if args.grid_out_dir is not None
            else grid_results / "_plots"
        )
        datasets = [d.strip() for d in args.grid_datasets.split(",") if d.strip()]
        stages = [s.strip() for s in args.grid_stages.split(",") if s.strip()]
        tag = "_".join(datasets)
        _plot_ablation_grid(
            grid_results, datasets, stages,
            grid_out_dir / f"{tag}_mvivf_ablation_grid.pdf",
            transpose=False,
        )
        _plot_ablation_grid(
            grid_results, datasets, stages,
            grid_out_dir / f"{tag}_mvivf_ablation_grid_transpose.pdf",
            transpose=True,
        )
        return 0

    if args.results is None:
        p.error("--results is required unless --grid is passed.")
    df = load_csvs(args.results)
    out_dir = args.out_dir if args.out_dir is not None else args.results
    plot(df, args.group_by, out_dir, prefix=args.prefix,
         extra_label_col=args.label_by,
         dataset=args.dataset, method_label=args.method_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
