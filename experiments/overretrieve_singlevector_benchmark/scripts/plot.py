#!/usr/bin/env python3
"""Plot Recall@k vs candidate budget k' from bench_singlevector_overretrieve logs.

The binary scans ``experiments/overretrieve_singlevector_benchmark/results/chamfer_<dataset>.txt``
(or any ``*.txt`` under ``--results-dir``). Each ``=== <method> ===`` section contains a table::

        k'        Recall@10
----------------------------------------
        10         0.8828
        ...

**Semantics** (see ``microbenchmark/quantization/bench_singlevector_overretrieve.cpp``): ground
truth is exact k-NN on the base points; for each query an approximate distance ranks all base
points; ``Recall@k`` at budget ``k'`` is the average fraction of the k true neighbors that appear
among the first ``k'`` positions of that approximate ranking. So the x-axis is over-retrieval
depth ``k'``, not the evaluation cutoff ``k``.

Visual style matches latency/multi-latency Pareto PDFs
(``experiments/builds/scripts/plot_stage.py``): white figure, light grid,
``tab10`` colors, and compact legend labels.

Usage::

    experiments/overretrieve_singlevector_benchmark/scripts/plot.py
    experiments/overretrieve_singlevector_benchmark/scripts/plot.py \\
        --datasets arguana,nfcorpus \\
        --results-dir experiments/overretrieve_singlevector_benchmark/results \\
        --out-dir experiments/overretrieve_singlevector_benchmark/results/_plots
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("pip install matplotlib numpy")

SECTION_HDR = re.compile(r"^===\s+(.+?)\s+===\s*$")
TABLE_HEADER = re.compile(r"k'\s+Recall@(\d+)")
DATA_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+(?:[eE][+-]?\d+)?)\s*$")


# Canonical section labels from bench_singlevector_overretrieve.cpp -> legend text.
_LEGEND_ABBREV: dict[str, str] = {
    "TurboQuant": "TQ-4bit",
    "TurboQuant-8bit": "TQ-8bit",
    "FastScan-b2": "FS-b2",
    "FastScan-b4": "FS-b4",
    "FastScan-b8": "FS-b8",
    "1BTQ": "TQ-1bit",
    "1BTQAsym": "TQ-1bit-asym",
    "Ref1BTQAsym": "Ref-TQ-1bit-asym",
    "Ref1BTQSym": "Ref-TQ-1bit-sym",
    "RaBitQ-1bit": "RQ-1bit",
    "RaBitQ-4bit": "RQ-4bit",
    "RaBitQ-8bit": "RQ-8bit",
}


def _legend_label(canonical: str) -> str:
    return _LEGEND_ABBREV.get(canonical, canonical)


def parse_benchmark_txt(text: str) -> tuple[int | None, dict[str, np.ndarray]]:
    """Return evaluation k and mapping method_label -> structured array with columns k_prime, recall."""
    k_eval: int | None = None
    for line in text.splitlines():
        if line.startswith("N=") and " k=" in line:
            m = re.search(r"\bk=(\d+)", line)
            if m:
                k_eval = int(m.group(1))
            break

    sections: list[tuple[str, list[tuple[int, float]]]] = []
    cur_label: str | None = None
    cur_rows: list[tuple[int, float]] = []
    pending_header = False

    for line in text.splitlines():
        msec = SECTION_HDR.match(line)
        if msec:
            if cur_label is not None:
                sections.append((cur_label, cur_rows))
            cur_label = msec.group(1).strip()
            cur_rows = []
            pending_header = False
            continue

        if cur_label is None:
            continue

        mh = TABLE_HEADER.search(line)
        if mh:
            k_eval = int(mh.group(1))
            pending_header = True
            continue

        if pending_header and line.strip().startswith("--"):
            continue

        mr = DATA_ROW.match(line)
        if mr and cur_label:
            k_prime = int(mr.group(1))
            recall = float(mr.group(2))
            cur_rows.append((k_prime, recall))

    if cur_label is not None:
        sections.append((cur_label, cur_rows))

    series: dict[str, np.ndarray] = {}
    for label, rows in sections:
        if not rows:
            continue
        arr = np.array(rows, dtype=float)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        series[label] = arr

    return k_eval, series


def _default_method_order() -> list[str]:
    """Stable legend order when present (matches ``All`` sweep in the binary)."""
    return [
        "TurboQuant",
        "TurboQuant-8bit",
        "FastScan-b2",
        "FastScan-b4",
        "FastScan-b8",
        "1BTQ",
        "1BTQAsym",
        "Ref1BTQAsym",
        "Ref1BTQSym",
        "RaBitQ-1bit",
        "RaBitQ-4bit",
        "RaBitQ-8bit",
    ]


def plot_dataset(
    dataset_title: str,
    k_eval: int | None,
    series: dict[str, np.ndarray],
    out_pdf: pathlib.Path,
) -> None:
    if not series:
        print(f"  [skip {dataset_title}] no parsed curves")
        return

    order = _default_method_order()
    labels = [lbl for lbl in order if lbl in series]
    labels += sorted(lbl for lbl in series if lbl not in labels)

    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")

    k_str = str(k_eval) if k_eval is not None else "?"

    for i, lbl in enumerate(labels):
        arr = series[lbl]
        kp = arr[:, 0]
        rec = arr[:, 1]
        color = cmap(i % 10)
        ax.plot(
            kp,
            rec,
            color=color,
            linewidth=1.9,
            label=_legend_label(lbl),
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Candidate budget $k'$")
    ax.set_ylabel(f"Recall@{k_str}")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)

    ax.set_title(f"{dataset_title}: Recall@{k_str} vs $k'$", fontsize=11)

    ncol = 2 if len(labels) > 5 else 1
    ax.legend(loc="best", ncol=ncol, fontsize=8)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def _discover_txt_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(results_dir.glob("chamfer_*.txt"))
    if files:
        return files
    return sorted(p for p in results_dir.glob("*.txt") if p.name != "summary.txt")


def _dataset_name_from_path(p: pathlib.Path) -> str:
    stem = p.stem
    if stem.startswith("chamfer_"):
        return stem[len("chamfer_") :]
    return stem


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    default_results = here.parent / "results"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=default_results,
        help="directory containing chamfer_<dataset>.txt logs",
    )
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="default: <results-dir>/_plots",
    )
    ap.add_argument(
        "--datasets",
        default=None,
        help="comma-separated dataset keys (e.g. arguana,nfcorpus); default: all chamfer_*.txt",
    )
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir else results_dir / "_plots"

    if args.datasets:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        paths = []
        for n in names:
            cand = results_dir / f"chamfer_{n}.txt"
            if cand.is_file():
                paths.append(cand)
            else:
                alt = results_dir / f"{n}.txt"
                if alt.is_file():
                    paths.append(alt)
                else:
                    print(f"[warn] missing results for dataset '{n}' under {results_dir}")
        if not paths:
            return 1
    else:
        paths = _discover_txt_files(results_dir)
        if not paths:
            print(f"[warn] no chamfer_*.txt under {results_dir}")
            return 0

    for path in paths:
        title = _dataset_name_from_path(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[skip {path}] {e}")
            continue
        k_eval, series = parse_benchmark_txt(text)
        out_pdf = out_dir / f"{title}_recall_vs_kprime.pdf"
        plot_dataset(title, k_eval, series, out_pdf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
