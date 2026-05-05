#!/usr/bin/env python3
"""Plot Recall@k vs candidate budget k' from bench_chamfer_overretrieve logs.

Reads ``chamfer_<dataset>.txt`` files from ``--results-dir`` and writes one PDF
per dataset under ``--out-dir``. Visual style matches latency/multi-latency
Pareto PDFs (white background, tab10, light grid).
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


HEADER_KP = re.compile(r"k'=(\d+)")
ROW_NUM = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

_ORDER = [
    "TurboQuant_mv (4bit)",
    "TurboQuant_mv (8bit)",
    "FastScan-b2",
    "FastScan-b4",
    "FastScan-b8",
    "RaBitQ-1bit",
    "RaBitQ-4bit",
    "RaBitQ-8bit",
    "1BTQ-mv",
    "1BTQAsym-mv",
]
_LABEL = {
    "TurboQuant_mv (4bit)": "TQ-MV-4bit",
    "TurboQuant_mv (8bit)": "TQ-MV-8bit",
    "FastScan-b2": "FS-b2",
    "FastScan-b4": "FS-b4",
    "FastScan-b8": "FS-b8",
    "RaBitQ-1bit": "RQ-1bit",
    "RaBitQ-4bit": "RQ-4bit",
    "RaBitQ-8bit": "RQ-8bit",
    "1BTQ-mv": "1BTQ-MV",
    "1BTQAsym-mv": "1BTQ-MV-Asym",
}


def _parse_table(text: str) -> tuple[int | None, dict[str, np.ndarray]]:
    k_eval = None
    m = re.search(r"Recall@(\d+)\s+vs candidate budget k'", text)
    if m:
        k_eval = int(m.group(1))

    lines = text.splitlines()
    header_idx = None
    kprimes: list[int] = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Method") and "k'=" in ln:
            header_idx = i
            kprimes = [int(x) for x in HEADER_KP.findall(ln)]
            break
    if header_idx is None or not kprimes:
        return k_eval, {}

    series: dict[str, np.ndarray] = {}
    for ln in lines[header_idx + 1 :]:
        s = ln.rstrip()
        if not s or s.startswith("==="):
            break
        if set(s.replace(" ", "")) == {"-"}:
            continue
        if len(s) < 24:
            continue
        method = s[:24].strip()
        vals = s[24:].split()
        if len(vals) != len(kprimes):
            continue
        if not all(ROW_NUM.match(v) for v in vals):
            continue
        rec = np.array([float(v) for v in vals], dtype=float)
        kp = np.array(kprimes, dtype=float)
        series[method] = np.column_stack([kp, rec])
    return k_eval, series


def _plot_dataset(dataset: str, k_eval: int | None, series: dict[str, np.ndarray], out_pdf: pathlib.Path) -> None:
    if not series:
        print(f"  [skip {dataset}] no parsed curves")
        return

    labels = [m for m in _ORDER if m in series] + sorted(m for m in series if m not in _ORDER)
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    k_str = str(k_eval) if k_eval is not None else "?"
    for i, method in enumerate(labels):
        arr = series[method]
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=cmap(i % 10),
            linewidth=1.9,
            label=_LABEL.get(method, method),
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Candidate budget $k'$")
    ax.set_ylabel(f"Recall@{k_str}")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.set_title(f"{dataset}: Recall@{k_str} vs $k'$", fontsize=11)
    ncol = 3 if len(labels) > 8 else (2 if len(labels) > 5 else 1)
    ax.legend(loc="best", ncol=ncol, fontsize=8)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def _dataset_name_from_path(p: pathlib.Path) -> str:
    stem = p.stem
    return stem[len("chamfer_") :] if stem.startswith("chamfer_") else stem


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    default_results = here.parent / "results"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=pathlib.Path, default=default_results)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--datasets", default=None, help="comma-separated dataset names")
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir else results_dir / "_plots"

    if args.datasets:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        paths = [results_dir / f"chamfer_{n}.txt" for n in names]
    else:
        paths = sorted(results_dir.glob("chamfer_*.txt"))

    found_any = False
    for p in paths:
        if not p.is_file():
            print(f"[warn] missing: {p}")
            continue
        found_any = True
        text = p.read_text(encoding="utf-8", errors="replace")
        k_eval, series = _parse_table(text)
        ds = _dataset_name_from_path(p)
        _plot_dataset(ds, k_eval, series, out_dir / f"{ds}_recall_vs_kprime.pdf")

    if not found_any:
        print(f"[warn] no chamfer_*.txt under {results_dir}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
