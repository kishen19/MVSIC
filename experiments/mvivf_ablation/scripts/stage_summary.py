#!/usr/bin/env python3
"""Stage-summary utility for MVIVF ablations.

Default behavior:
1) Generate stage plots (delegates to plot.py).
2) Compute tree-quality for each build_config present in the stage results.
3) Write tree-quality tables.

Optional behavior:
- Compute budget-based winner tables (disabled by default).

Example:
    python3 experiments/mvivf_ablation/scripts/stage_summary.py \
      --results experiments/mvivf_ablation/results/nq500k/mvivf \
      --indices-root experiments/mvivf_ablation/indices/nq500k/mvivf \
      --db data/beir/nq500k/nq500k_points.pcs \
      --queries data/beir/nq500k/nq500k_queries.pcs \
      --gt data/beir/nq500k/nq500k_chamfer_neighbors.gt \
      --group-by k_per_level \
      --stage-label stage1

To also compute budget winners:
    ... --with-budget-winners --budgets 0.5,1.0,2.0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Iterable

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def _load_plot_module():
    # Local sibling script import.
    sys.path.insert(0, str(HERE))
    import plot  # pylint: disable=import-error,import-outside-toplevel

    return plot


def _parse_budgets(s: str) -> list[float]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("No budgets parsed; pass comma-separated floats, e.g. 0.5,1,2")
    return vals


def _select_winners(df: pd.DataFrame, group_col: str, budgets_ms: Iterable[float]) -> pd.DataFrame:
    rows = []
    for b in budgets_ms:
        cand = df[df["latency_ms"] <= b].copy()
        if cand.empty:
            rows.append(
                {
                    "budget_ms": b,
                    "status": "no_feasible_points",
                }
            )
            continue

        # Per group: best recall under budget, then lowest latency, then lowest avg_cmps.
        per_group = (
            cand.sort_values(
                by=[group_col, "recall_k_k", "latency_ms", "avg_cmps"],
                ascending=[True, False, True, True],
            )
            .groupby(group_col, as_index=False)
            .first()
        )

        winner = per_group.sort_values(
            by=["recall_k_k", "latency_ms", "avg_cmps"],
            ascending=[False, True, True],
        ).iloc[0]

        row = {"budget_ms": b, "status": "ok"}
        for c in [
            group_col,
            "build_config",
            "variant_name",
            "search_name",
            "k",
            "nprobes",
            "num_rerank",
            "recall_1_k",
            "recall_k_k",
            "latency_ms",
            "avg_cmps",
            "QPS_seq",
        ]:
            if c in winner.index:
                row[c] = winner[c]
        rows.append(row)
    return pd.DataFrame(rows)


def _run_tree_quality(
    tree_quality_script: pathlib.Path,
    index_path: pathlib.Path,
    db: pathlib.Path,
    queries: pathlib.Path,
    gt: pathlib.Path,
    k: int,
    metric: str,
    out_json: pathlib.Path,
    family: str,
) -> dict:
    cmd = [
        sys.executable,
        str(tree_quality_script),
        "--index",
        str(index_path),
        "--db",
        str(db),
        "--queries",
        str(queries),
        "--gt",
        str(gt),
        "--k",
        str(k),
        "--metric",
        metric,
        "--family",
        family,
        "--out",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text())


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute stage plots + tree quality (+ optional winners).")
    ap.add_argument("--results", type=pathlib.Path, required=True,
                    help="Results root (e.g. experiments/.../results/nq500k/mvivf)")
    ap.add_argument("--indices-root", type=pathlib.Path, required=True,
                    help="Indices root for this method (e.g. experiments/.../indices/nq500k/mvivf)")
    ap.add_argument("--db", type=pathlib.Path, required=True)
    ap.add_argument("--queries", type=pathlib.Path, required=True)
    ap.add_argument("--gt", type=pathlib.Path, required=True)
    ap.add_argument("--group-by", default="k_per_level")
    ap.add_argument("--stage-label", default="stage1",
                    help="Output prefix for plots/tables, e.g. stage1, stage2, ...")
    ap.add_argument("--budgets", default="0.5,1.0,2.0")
    ap.add_argument("--metric", choices=("ip", "l2"), default="ip")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None,
                    help="Output directory for stage tables/plots (default: --results)")
    ap.add_argument("--plot-script", type=pathlib.Path, default=HERE / "plot.py")
    ap.add_argument("--tree-quality-script", type=pathlib.Path, default=HERE / "tree_quality.py")
    ap.add_argument("--skip-tree-quality", action="store_true",
                    help="Skip tree-quality computation.")
    ap.add_argument("--with-budget-winners", action="store_true",
                    help="Also compute budget-based winner tables.")
    args = ap.parse_args()
    family = "mvivf"
    ir_name = args.indices_root.name.lower()
    if "spill" in ir_name:
        family = "mvivf_spill"
    elif "flat" in ir_name:
        family = "mvivf_flat"


    out_dir = args.out_dir if args.out_dir is not None else args.results
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build plots + load merged dataframe through plot.py.
    plot_mod = _load_plot_module()
    df = plot_mod.load_csvs(args.results)
    # plot_mod.plot() handles group-column synthesis and writes <stage-label>_*.pdf.
    plot_mod.plot(df, args.group_by, out_dir, prefix=args.stage_label)

    # Recompute latency here so winner selection has explicit numeric field.
    if "QPS_seq" not in df.columns:
        raise SystemExit("Missing QPS_seq in merged CSVs.")
    df = df.assign(latency_ms=1000.0 / df["QPS_seq"])
    df = plot_mod._ensure_group_col(df, args.group_by)  # noqa: SLF001

    # 2) Tree quality for all build configs in this stage.
    tree_stats_by_build = {}
    if not args.skip_tree_quality and "build_config" in df.columns:
        if family != "mvivf":
            print(
                f"[warn] tree_quality currently supports only regular mvivf, "
                f"but family={family}; skipping tree-quality for this run."
            )
            build_cfgs = []
        else:
            build_cfgs = sorted(set(df["build_config"].dropna().astype(str)))
        for build_cfg in build_cfgs:
            index_path = args.indices_root / build_cfg / "index.bin"
            if not index_path.exists():
                print(f"[warn] missing index for tree-quality: {index_path}")
                continue
            out_json = out_dir / f"tree_quality_{build_cfg}.json"
            try:
                tree_stats_by_build[build_cfg] = _run_tree_quality(
                    tree_quality_script=args.tree_quality_script,
                    index_path=index_path,
                    db=args.db,
                    queries=args.queries,
                    gt=args.gt,
                    k=args.k,
                    metric=args.metric,
                    out_json=out_json,
                    family=family,
                )
            except subprocess.CalledProcessError as e:
                print(f"[warn] tree_quality failed for {build_cfg}: {e}")

    tree_rows = []
    for b in sorted(tree_stats_by_build):
        row = {"build_config": b}
        row.update(tree_stats_by_build[b])
        tree_rows.append(row)
    tree_df = pd.DataFrame(tree_rows)
    tree_csv = out_dir / f"{args.stage_label}_tree_quality.csv"
    tree_md = out_dir / f"{args.stage_label}_tree_quality.md"
    if not tree_df.empty:
        tree_df.to_csv(tree_csv, index=False)
        tree_md.write_text(_to_markdown_table(tree_df))
        print(f"Wrote {tree_csv}")
        print(f"Wrote {tree_md}")

    # 3) Optional budget-based winner table.
    if args.with_budget_winners:
        budgets = _parse_budgets(args.budgets)
        winners = _select_winners(df, args.group_by, budgets)
        if tree_stats_by_build and "build_config" in winners.columns:
            for key in [
                "num_internal_nodes",
                "num_leaves",
                "height",
                "leaf_avg",
                "leaf_median",
                "root_child_avg",
                "greedy_leaf_cover_avg",
                "best_leaf_count_avg",
            ]:
                winners[key] = winners["build_config"].map(
                    lambda b: tree_stats_by_build.get(str(b), {}).get(key)
                )

        winners_csv = out_dir / f"{args.stage_label}_winners.csv"
        winners_md = out_dir / f"{args.stage_label}_winners.md"
        winners.to_csv(winners_csv, index=False)
        winners_md.write_text(_to_markdown_table(winners))
        print(f"Wrote {winners_csv}")
        print(f"Wrote {winners_md}")

    print(f"Plots: {out_dir / (args.stage_label + '_latency.pdf')}")
    print(f"Plots: {out_dir / (args.stage_label + '_cmps.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

