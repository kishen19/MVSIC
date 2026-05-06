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

# Stable plot/legend order. Top-to-bottom: TQ -> RQ -> FS, so in the
# default single-column legend the four TQ rows form a contiguous block at
# the top, three RQ rows in the middle, three FS rows at the bottom.
# Mirrors _default_method_order() in the SV plot so corresponding methods
# sit at the same legend position across the SV and MV PDFs.
_ORDER = [
    # TurboQuant family (TQ-1bit, TQ-1bit-asym, TQ-4bit, TQ-8bit)
    "1BTQ-mv",
    "1BTQAsym-mv",
    "TurboQuant_mv (4bit)",
    "TurboQuant_mv (8bit)",
    # RaBitQ family (RQ-1bit, RQ-4bit, RQ-8bit)
    "RaBitQ-1bit",
    "RaBitQ-4bit",
    "RaBitQ-8bit",
    # FastScan family (FS-b2, FS-b4, FS-b8)
    "FastScan-b2",
    "FastScan-b4",
    "FastScan-b8",
]
# Display labels shown in the legend. Kept identical to the SV plot's
# _LEGEND_ABBREV (modulo the SV-only Ref* rows) so the MV PDF and the SV PDF
# read with the same legend across the panel — a "TQ-1bit" line in one is
# "TQ-1bit" in the other, plotted in the same color.
_LABEL = {
    "TurboQuant_mv (4bit)": "TQ-4bit",
    "TurboQuant_mv (8bit)": "TQ-8bit",
    "FastScan-b2":          "FS-b2",
    "FastScan-b4":          "FS-b4",
    "FastScan-b8":          "FS-b8",
    "RaBitQ-1bit":          "RQ-1bit",
    "RaBitQ-4bit":          "RQ-4bit",
    "RaBitQ-8bit":          "RQ-8bit",
    "1BTQ-mv":              "TQ-1bit",
    "1BTQAsym-mv":          "TQ-1bit-asym",
}

# Canonical "method family" for cross-experiment color sharing. The SV plot in
# experiments/overretrieve_singlevector_benchmark/scripts/plot.py uses the same
# family -> color map so corresponding rows (TQ-4bit / TQ-MV-4bit, FS-b8 /
# FS-b8, 1BTQ / 1BTQ-MV, ...) share a color across the SV and MV PDFs. Keep
# the keys here in sync with that file's _SECTION_TO_FAMILY.
_SECTION_TO_FAMILY: dict[str, str] = {
    "TurboQuant_mv (4bit)": "TQ-4bit",
    "TurboQuant_mv (8bit)": "TQ-8bit",
    "FastScan-b2":          "FS-b2",
    "FastScan-b4":          "FS-b4",
    "FastScan-b8":          "FS-b8",
    "RaBitQ-1bit":          "RQ-1bit",
    "RaBitQ-4bit":          "RQ-4bit",
    "RaBitQ-8bit":          "RQ-8bit",
    "1BTQ-mv":              "1BTQ",
    "1BTQAsym-mv":          "1BTQAsym",
}

_FAMILY_COLORS: dict[str, str] = {
    "TQ-4bit":  "tab:blue",
    "TQ-8bit":  "tab:orange",
    "FS-b2":    "tab:green",
    "FS-b4":    "tab:red",
    "FS-b8":    "tab:purple",
    "RQ-1bit":  "tab:brown",
    "RQ-4bit":  "tab:pink",
    "RQ-8bit":  "tab:gray",
    "1BTQ":     "tab:olive",
    "1BTQAsym": "tab:cyan",
}


def _color_for(method: str, fallback_idx: int):
    """Map an MV section name to its shared cross-experiment color.

    Anything we don't recognise falls back to tab10 cycling at ``fallback_idx``
    so a new method auto-gets a distinct color.
    """
    fam = _SECTION_TO_FAMILY.get(method)
    if fam is not None:
        c = _FAMILY_COLORS.get(fam)
        if c is not None:
            return c
    cmap = plt.colormaps.get_cmap("tab10")
    return cmap(fallback_idx % 10)


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

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    k_str = str(k_eval) if k_eval is not None else "?"
    for i, method in enumerate(labels):
        arr = series[method]
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=_color_for(method, i),
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
