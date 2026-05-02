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
# long names used as the `--group-by` argument; values are regexes matched
# against `build_config` (or fallback path strings) with a single capture
# group for the integer value.
#
# Build-config naming convention (mvivf):
#     mvivf_k<K>_l<L>_d<D>_n<N>_s<S>_mpcc<M>_mpcik<MK>
# Only the suffixes relevant to the current stage need to be present; missing
# suffixes leave the corresponding derived column NaN.
#
# For mvivf_spill we also support `_s<N>` -> num_spill (same regex as `s`;
# choose whichever column is meaningful for the family being plotted).
_KNOB_PATTERNS: dict[str, str] = {
    "k_per_level": r"_k(\d+)(?:_|$)",
    "max_leaf_size": r"_l(\d+)(?:_|$)",
    "max_depth": r"_d(\d+)(?:_|$)",
    "niters": r"_nit(\d+)(?:_|$)",
    "s": r"_s(\d+)(?:_|$)",
    "max_point_clouds_per_cluster": r"_mpcc(\d+)(?:_|$)",
    "max_points_per_centroid_inner_kmeans": r"_mpcik(\d+)(?:_|$)",
    # mvivf_spill aliases (same regex as `s`, exposed under a different name).
    "num_spill": r"_s(\d+)(?:_|$)",
    "num_spill_l2": r"_l2s(\d+)(?:_|$)",
}


def _derive_knob_column(series: pd.Series, pattern: str) -> pd.Series:
    extracted = series.astype(str).str.extract(pattern)[0]
    return extracted.apply(
        lambda x: int(x) if isinstance(x, str) and x.isdigit() else x
    )


def _populate_knobs(df: pd.DataFrame, source_col: str = "build_config") -> pd.DataFrame:
    if source_col not in df.columns:
        return df
    for col, pat in _KNOB_PATTERNS.items():
        if col in df.columns:
            continue
        df[col] = _derive_knob_column(df[source_col], pat)
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
    if group_col in df.columns and df[group_col].notna().any():
        return df
    # Try to derive from build_config / source paths using the canonical
    # short-name convention (see _KNOB_PATTERNS above).
    pattern = _KNOB_PATTERNS.get(group_col)
    if pattern is not None:
        for src_col in ("build_config", "source_relpath", "source_abspath"):
            if src_col not in df.columns:
                continue
            derived = _derive_knob_column(df[src_col], pattern)
            if derived.notna().any():
                df[group_col] = derived
                break
    if group_col not in df.columns or not df[group_col].notna().any():
        raise SystemExit(
            f"group column '{group_col}' not found and could not be derived "
            f"from build_config. Available columns: {sorted(df.columns)}"
        )
    return df


def _plot_xy(df: pd.DataFrame, group_col: str, x_col: str, y_col: str,
             x_label: str, y_label: str, title: str, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, sub in df.groupby(group_col):
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], marker="o", label=f"{group_col}={key}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
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


def _plot_qps_pareto(df: pd.DataFrame, group_col: str, out_path: pathlib.Path, title: str) -> None:
    if "QPS_seq" not in df.columns:
        raise SystemExit("Missing QPS_seq column in merged CSVs.")
    if "recall_k_k" not in df.columns:
        raise SystemExit("Missing recall_k_k column in merged CSVs.")

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, sub in df.groupby(group_col):
        sub = sub.dropna(subset=["recall_k_k", "QPS_seq"])
        if sub.empty:
            continue
        ax.scatter(sub["recall_k_k"], sub["QPS_seq"], s=16, alpha=0.35)
        fx, fy = _pareto_curve(sub["recall_k_k"], sub["QPS_seq"])
        if len(fx) == 0:
            continue
        ax.plot(fx, fy, marker="o", label=f"{group_col}={key}")

    ax.set_xlabel("Recall@k (k@k)")
    ax.set_ylabel("QPS (per-query)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")
    plt.close(fig)


def plot(df: pd.DataFrame, group_col: str, out_dir: pathlib.Path, prefix: str = "stage1") -> None:
    if "QPS_seq" in df.columns:
        df = df.assign(latency_ms=1000.0 / df["QPS_seq"])
    else:
        raise SystemExit("Missing QPS_seq column in merged CSVs.")
    if "avg_cmps" not in df.columns:
        raise SystemExit("Missing avg_cmps column in merged CSVs.")
    df = _ensure_group_col(df, group_col)

    # For stage selection we use Recall@k (k@k).
    recall_x = "recall_k_k"
    if recall_x not in df.columns:
        raise SystemExit("Missing recall_k_k column in merged CSVs.")

    _plot_qps_pareto(
        df,
        group_col=group_col,
        out_path=out_dir / f"{prefix}_pareto.pdf",
        title=f"{prefix}: QPS vs recall Pareto (grouped by {group_col})",
    )
    _plot_xy(
        df,
        group_col=group_col,
        x_col=recall_x,
        y_col="latency_ms",
        x_label="Recall@k (k@k)",
        y_label="Sequential latency per query (ms)",
        title=f"{prefix}: latency vs recall (grouped by {group_col})",
        out_path=out_dir / f"{prefix}_latency.pdf",
    )
    _plot_xy(
        df,
        group_col=group_col,
        x_col=recall_x,
        y_col="avg_cmps",
        x_label="Recall@k (k@k)",
        y_label="Average bytes accessed per query",
        title=f"{prefix}: bytes accessed vs recall (grouped by {group_col})",
        out_path=out_dir / f"{prefix}_cmps.pdf",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=pathlib.Path, required=True)
    p.add_argument("--group-by", default="build_config",
                   help="CSV column to group curves by.")
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
    args = p.parse_args()
    df = load_csvs(args.results)
    out_dir = args.out_dir if args.out_dir is not None else args.results
    plot(df, args.group_by, out_dir, prefix=args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
