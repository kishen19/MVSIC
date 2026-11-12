"""Installed plotting for the existing ``benchmarks/plot_configs`` YAML files.

Keep the result layout, Pareto smoothing, interpolation, and output names from
``benchmarks/plot.py``. CSV parsing needs no pandas/seaborn dependency; matplotlib
is imported only when drawing a plot. Paths remain relative to the caller's
working directory, as in the original benchmark scripts.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .config import read_config


def is_plot_config(raw):
    return isinstance(raw, dict) and ("datasets" in raw or "dataset" in raw) and (
        "plots" in raw or "results" in raw)


def _string(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{name} must be a {'possibly empty' if empty else 'nonempty'} string")
    return value


def validate_plot_config(raw, base_dir=None):
    """Validate both existing YAML forms without opening results or plotting."""
    if not is_plot_config(raw):
        raise ValueError("plot config requires dataset(s) and plots or results")
    root = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
    name = _string(raw.get("name", "plots"), "name")
    datasets = raw.get("datasets") or [raw.get("dataset")]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a nonempty list")
    resolved_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("each dataset must be a mapping")
        label = _string(dataset.get("name"), "dataset.name")
        results = Path(_string(dataset.get("results"), "dataset.results")).expanduser()
        resolved_datasets.append({"name": label, "results": root / results})

    plots = raw.get("plots", raw.get("results"))
    if not isinstance(plots, list) or not plots:
        raise ValueError("plots/results must be a nonempty list")
    if all(isinstance(entry, (list, tuple)) for entry in plots):
        plots = [{"results": plots}]
    groups = []
    for group in plots:
        if not isinstance(group, dict):
            raise ValueError("plot groups must be mappings or result lists")
        group_name = _string(group.get("name", ""), "plot.name", empty=True)
        k = group.get("k") if group.get("k") is not None else raw.get("k", 10)
        if type(k) is not int or k <= 0:
            raise ValueError("plot k must be a positive integer")
        entries = group.get("results")
        if not isinstance(entries, list) or not entries:
            raise ValueError("each plot group requires a nonempty results list")
        checked = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError("a result entry must be [method, build, variant, ...]")
            method = _string(entry[0], "result method")
            build = _string(entry[1], "result build")
            variants = [_string(v, "result variant", empty=True) for v in entry[2:]] or [""]
            checked.append([method, build, *variants])
        groups.append({"name": group_name, "k": k, "results": checked})
    return {"name": name, "datasets": resolved_datasets, "groups": groups,
            "output": root / f"results/{name}"}


def _number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def calculate_pareto(rows, x_col, y_col, min_recall_spacing=0.01):
    """Return ascending (recall, QPS) points with the original smoothing rule."""
    best = {}
    for row in rows:
        x, y = _number(row.get(x_col)), _number(row.get(y_col))
        # Missing measurements are not zero throughput. Nonpositive QPS also
        # cannot be represented on the original logarithmic plot axes.
        if x is not None and y is not None and y > 0:
            best[x] = max(best.get(x, -math.inf), y)
    front, maximum = [], -math.inf
    for x, y in sorted(best.items(), reverse=True):
        if y > maximum:
            front.append((x, y))
            maximum = y
    if not front:
        return []
    simplified = [front[0]]
    for point in front[1:]:
        if abs(point[0] - simplified[-1][0]) > min_recall_spacing:
            simplified.append(point)
    if simplified[-1][0] != front[-1][0]:
        if abs(front[-1][0] - simplified[-1][0]) <= min_recall_spacing:
            simplified[-1] = front[-1]
        else:
            simplified.append(front[-1])
    return sorted(simplified)


def interpolate_qps(front, target):
    """Interpolate linearly only between measured frontier points."""
    for x, y in front:
        if x == target:
            return y
    for (x1, y1), (x2, y2) in zip(front, front[1:]):
        if x1 < target < x2:
            return y1 + (y2 - y1) * (target - x1) / (x2 - x1)
    return None


def _join_affixes(*parts):
    return "_".join(part for part in parts if part)


def _result_files(directory, k, variant):
    # The older plot schema names search variants in one flat directory.
    # Preserve its explicit file selection when that file exists.
    search = directory / f"k={k}"
    flat = search / (f"{variant}_results.csv" if variant else "results.csv")
    if flat.is_file():
        if flat.name.startswith("multi_latency_results"):
            return []
        mode = ("latency" if flat.name.startswith("latency_results") else
                "batch" if flat.name.startswith("batch_results") else None)
        return [(flat, mode)]

    # benchmark_search.py separates serving variants into directories and
    # timing modes into filenames. A variant without its own directory can
    # still select a named search group, e.g. latency_results_nr50.csv.
    serving = directory / variant / f"k={k}" if variant else search
    named_search = bool(variant) and not serving.is_dir()
    if named_search:
        serving = search
    found = []
    for prefix, mode in (("results", None), ("latency_results", "latency"), ("batch_results", "batch")):
        candidates = ([serving / f"{prefix}_{variant}.csv"] if named_search else
                      [serving / f"{prefix}.csv", *sorted(serving.glob(f"{prefix}_*.csv"))])
        found.extend((path, mode) for path in candidates if path.is_file())
    # multi_latency_results*.csv deliberately does not match: its QPS_par
    # measures parallel single-query calls, not batched search throughput.
    return found


def _load_series(dataset, group):
    series = []
    for position, (method, build, *variants) in enumerate(group["results"]):
        base = build if build.startswith(method) else f"{method}_{build}"
        label = f"{base} (Pareto)" if len(variants) > 1 else _join_affixes(base, variants[0])
        directory = dataset["results"] / method / build
        rows = []
        seen = set()
        for variant in variants:
            files = _result_files(directory, group["k"], variant)
            if not files:
                print(f'  Warning: No result CSV for {directory}, k={group["k"]}, variant={variant!r}')
                continue
            for path, mode in files:
                if path in seen:
                    continue
                seen.add(path)
                with path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if mode == "latency":
                            row["QPS_par"] = ""
                        elif mode == "batch":
                            row["QPS_seq"] = ""
                        rows.append(row)
        if not rows:
            continue
        build_time = None
        stats = directory / "build_stats.json"
        if stats.is_file():
            try:
                value = json.loads(stats.read_text())
                if isinstance(value, dict):
                    build_time = _number(value.get("build_time_sec"))
            except (ValueError, TypeError):
                pass
        fronts = {(x, y): calculate_pareto(rows, x, y)
                  for x in ("recall_1_k", "recall_k_k") for y in ("QPS_seq", "QPS_par")}
        series.append({"method": method, "label": label, "position": position,
                       "fronts": fronts, "build_time_sec": build_time})
    return series


def _write_analysis(series, directory, dataset, k, affix):
    paths = []
    for target in (90, 95, 99):
        records = []
        for entry in series:
            row = {"Method": entry["label"], "build_time_sec": entry["build_time_sec"]}
            for x in ("recall_1_k", "recall_k_k"):
                for y in ("QPS_seq", "QPS_par"):
                    row[f"{y}_at_{target}_{x}"] = interpolate_qps(entry["fronts"][x, y], target / 100)
            records.append(row)
        if not records:
            continue
        # Retain the leading underscore when affix is empty, matching plot.py.
        path = directory / f"{affix}_{dataset}_k={k}_qps_at_{target}_recall.csv"
        directory.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            for row in records:
                writer.writerow({key: "" if value is None else f"{value:.2f}" if isinstance(value, float) else value
                                 for key, value in row.items()})
        print(f"Saving QPS @ {target}% Recall analysis to: {path}")
        paths.append(path)
    return paths


def _draw_plot(series, directory, dataset, k, affix, file_format, style):
    if not any(front for entry in series for front in entry["fronts"].values()):
        print("No measured QPS and recall found to plot.")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('plotting requires pip install "mvsic[plot]"') from error
    advanced = style == "advanced"
    settings = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                "axes.labelsize": 20 if advanced else 18, "font.size": 18 if advanced else 16,
                "legend.fontsize": 16 if advanced else 14, "xtick.labelsize": 16 if advanced else 14,
                "ytick.labelsize": 16 if advanced else 14, "lines.markersize": 10,
                "figure.titlesize": 24 if advanced else 22}
    palette = ["#023eff", "#ff7c00", "#1ac938", "#e8000b", "#8b2be2", "#9f4800", "#f14cc1", "#a3a3a3", "#ffc400", "#00d7ff"]
    colors = {"mvivf": "#023eff", "svh_ivf": "#1ac939", "muvera": "#8b2be2", "vamana": "#e8050a",
              "fastplaid": "#ffc402", "mvivf_flat": "#00d7ff", "svh": "#9f4801"}
    markers = {"mvivf": "o", "svh_ivf": "X", "muvera": "s", "vamana": "D", "fastplaid": "^", "mvivf_flat": "v", "svh": "<"}
    available = ["o", "s", "X", "D", "^", "v", "<", ">", "P", "*"]
    with plt.rc_context(settings):
        figure, axes = plt.subplots(2, 2, figsize=(20, 18))
        try:
            figure.suptitle(f"{dataset} ($k={k}$)", fontweight="bold")
            legend = {}
            for axis, (x, y) in zip(axes.flat, (("recall_1_k", "QPS_seq"), ("recall_k_k", "QPS_seq"),
                                                ("recall_1_k", "QPS_par"), ("recall_k_k", "QPS_par"))):
                recall = f"Recall 1@{k}" if x == "recall_1_k" else f"Recall {k}@{k}"
                axis.set_xlabel(recall)
                axis.set_ylabel("QPS (Log Scale)")
                axis.set_title(f'{"QPS (Batched)" if y == "QPS_par" else "QPS"} vs. {recall}', fontsize=20 if advanced else 15)
                axis.set_yscale("log")
                axis.grid(True, which="major", linestyle="-", alpha=0.6)
                axis.grid(True, which="minor", linestyle=":", alpha=0.3)
                axis.spines[["top", "right"]].set_visible(False)
                minima = []
                for entry in series:
                    front = entry["fronts"][x, y]
                    if not front:
                        continue
                    method, label, i = entry["method"], entry["label"], entry["position"]
                    color, marker = palette[i % len(palette)], available[i % len(available)]
                    line, width = ["-", "--", "-.", ":"][i % 4], 2.5
                    if advanced:
                        color = colors.setdefault(method, next((value for key, value in colors.items() if key in method), color))
                        marker = markers.setdefault(method, marker)
                        lower = label.lower()
                        line, width = ("--", 4) if "fastscan" in lower else (":", 5) if "rabitq" in lower else ("-.", 4) if "pq" in lower else ("-", 4)
                    artist, = axis.plot([p[0] for p in front], [p[1] for p in front], marker=marker,
                                        linestyle=line, linewidth=width, label=label, color=color,
                                        markeredgecolor="black", markeredgewidth=0.3, alpha=0.9)
                    legend.setdefault(label, artist)
                    minima.append(front[0][0])
                if len(minima) > 1:
                    axis.set_xlim(left=max(0, sorted(minima)[1] - 0.05), right=1.01)
            figure.legend(list(legend.values()), list(legend), loc="lower center", ncol=min(4, len(legend)),
                          bbox_to_anchor=(0.5, 0.01), frameon=True)
            figure.tight_layout(rect=[0, 0.06, 1, 0.95])
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{_join_affixes(affix, style)}_{dataset}_k={k}_pareto.{file_format}"
            figure.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saving Pareto plot to: {path}")
            return path
        finally:
            plt.close(figure)


def plot_config(filename_or_raw, *, affix="", style="default", file_format="png", base_dir=None):
    """Plot existing YAML/CSV layouts directly; never invent absent statistics."""
    raw = filename_or_raw if isinstance(filename_or_raw, dict) else read_config(Path(filename_or_raw))
    plan = validate_plot_config(raw, base_dir)
    _string(affix, "affix", empty=True)
    if style not in ("default", "advanced") or file_format not in ("png", "pdf"):
        raise ValueError("plot style must be default/advanced and format must be png/pdf")
    paths = []
    for dataset in plan["datasets"]:
        for group in plan["groups"]:
            series = _load_series(dataset, group)
            suffix = _join_affixes(affix, group["name"])
            path = _draw_plot(series, plan["output"], dataset["name"], group["k"], suffix, file_format, style)
            if path is not None:
                paths.append(path)
            paths.extend(_write_analysis(series, plan["output"], dataset["name"], group["k"], suffix))
    return paths
