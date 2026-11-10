#!/usr/bin/env python3
"""Validate and plot the native Vamana TQ4/OneBitTQ leaf ablation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


WIDTHS = (64, 128, 256, 512, 1024, 2048, 4096)
RERANKS = (0, 20, 40)
QUANTS = {"tq4": ("TQ4 leaf", "#2568a5"), "tq1": ("OneBitTQ leaf", "#c4641d")}
METRICS = ("recall_k_k", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "QPS_seq")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def load_experiment(base: Path, dataset: str, partial: bool) -> tuple[list[dict], dict]:
    settings_path = base / "settings.json"
    settings = read_json(settings_path, {})
    repeats = int(settings.get("repeats", 1))
    query_path = base / "query_ids.txt"
    query_hash = digest(query_path) if query_path.exists() else None
    query_count = len(query_path.read_bytes().splitlines()) if query_path.exists() else None
    model_hashes = {path.name: digest(path) for path in sorted(base.glob("shared_quantizer.tq*")) if path.is_file()}
    incomplete, snapshots, points = [], [], []
    if settings and (settings.get("early_stop") is not False or settings.get("latency_threads") != 1):
        raise ValueError("expected complete-grid, single-thread latency measurements")
    if settings and (query_hash != settings.get("query_ids_sha256") or query_count != settings.get("query_count")):
        raise ValueError("query-ID file differs from settings provenance")
    if not settings:
        incomplete.append("settings.json is not available")
    same_rotation = all(f"shared_quantizer.{quant}" in model_hashes for quant in QUANTS)
    if same_rotation:
        same_rotation = model_hashes["shared_quantizer.tq4"] == model_hashes["shared_quantizer.tq1"]
        if not same_rotation:
            raise ValueError("TQ4 and OneBitTQ leaf caches contain different rotators")
    else:
        incomplete.append("both leaf caches are not yet available")
    if "shared_quantizer.tq8" not in model_hashes:
        incomplete.append("shared TQ8 rerank cache is not yet available")
    graph_path = Path(settings["index_path"]) if settings.get("index_path") else None
    graph_hash = digest(graph_path) if graph_path and graph_path.is_file() else None
    settings_hash = digest(settings_path) if settings_path.exists() else None
    for quant in QUANTS:
        for rerank in RERANKS:
            path = base / "results" / quant / f"latency_results_nr{rerank}.csv"
            marker_path = path.with_suffix(".complete.json")
            marker = read_json(marker_path)
            snapshot = {"quantizer": quant, "num_rerank": rerank, "csv": str(path),
                        "status": "not_started", "rows": 0, "points": []}
            snapshots.append(snapshot)
            if not path.exists():
                incomplete.append(f"{quant}/nr{rerank}: no CSV")
                continue
            raw = path.read_bytes()
            complete = raw[:raw.rfind(b"\n") + 1]
            rows = list(csv.DictReader(io.StringIO(complete.decode("utf-8"))))
            snapshot.update(rows=len(rows), sha256=hashlib.sha256(raw).hexdigest(),
                            bytes=len(raw), ignored_partial_tail_bytes=len(raw) - len(complete), status="running")
            seen, grouped = set(), {}
            for row in rows:
                expected = {"method": "mv_vamana", "quant": "tq" if quant == "tq4" else "tq1",
                            "num_rerank": str(rerank), "tq8_rerank": "1", "k": "10", "R": "64",
                            "build_L": "128", "two_pass": "0", "query_seed": "42"}
                if any(row.get(key) != value for key, value in expected.items()) or float(row["alpha"]) != 1:
                    raise ValueError(f"unexpected benchmark settings in {path}")
                width, repeat = int(row["L"]), int(row["repeat"])
                if width not in WIDTHS or not 0 <= repeat < repeats or (repeat, width) in seen:
                    raise ValueError(f"duplicate/unexpected repeat or L in {path}")
                seen.add((repeat, width))
                if query_count is not None and int(row["num_queries"]) != query_count:
                    raise ValueError(f"query count differs in {path}")
                parsed = {key: float(row[key]) for key in METRICS}
                if any(not math.isfinite(item) for item in parsed.values()) or parsed["mean_latency_ms"] <= 0:
                    raise ValueError(f"invalid benchmark metric in {path}")
                if not math.isclose(1000 / parsed["QPS_seq"], parsed["mean_latency_ms"], rel_tol=1e-8):
                    raise ValueError(f"QPS/latency disagree in {path}")
                grouped.setdefault(width, []).append(parsed)
            sample_path = Path(str(path) + ".query_ids.txt")
            sample_hash = digest(sample_path) if sample_path.exists() else None
            sample_verified = sample_hash == query_hash and query_hash is not None
            snapshot.update(query_ids_sha256=sample_hash, query_sample_verified=sample_verified)
            expected_grid = {(repeat, width) for repeat in range(repeats) for width in WIDTHS}
            if marker:
                if snapshot["sha256"] != marker.get("csv_sha256"):
                    raise ValueError(f"completed CSV fingerprint changed: {path}")
                if marker.get("graph_sha256") != graph_hash or marker.get("settings_sha256") != settings_hash:
                    raise ValueError(f"completed curve graph/settings changed: {path}")
                for name, fingerprint in marker.get("models", {}).items():
                    if model_hashes.get(name) != fingerprint:
                        raise ValueError(f"quantizer changed after measuring {path}: {name}")
                if f"shared_quantizer.{quant}" not in marker.get("models", {}):
                    raise ValueError(f"leaf model fingerprint missing in {marker_path}")
                if rerank and "shared_quantizer.tq8" not in marker.get("models", {}):
                    raise ValueError(f"rerank model fingerprint missing in {marker_path}")
                command = marker["command"]
                if Path(command_value(command, "--index")).resolve() != graph_path.resolve():
                    raise ValueError(f"curve uses another graph: {path}")
                if Path(command_value(command, "--quantizer-cache-prefix")).resolve() != (base / "shared_quantizer").resolve():
                    raise ValueError(f"curve uses another quantizer cache: {path}")
                if "--early-stop" in command or command[:3] != ["taskset", "-c", str(settings["latency_cpu"])]:
                    raise ValueError(f"unexpected early-stop/CPU settings: {path}")
                if seen != expected_grid or len(rows) != repeats * len(WIDTHS) or not sample_verified:
                    raise ValueError(f"completed curve has incomplete grid or mismatched query IDs: {path}")
                snapshot["status"] = "complete"
                snapshot["completion_marker"] = marker
            else:
                incomplete.append(f"{quant}/nr{rerank}: completion marker pending")
            for width, measurements in sorted(grouped.items()):
                point = {"dataset": dataset, "quantizer": quant, "num_rerank": rerank, "L": width,
                         "k": 10, "num_queries": query_count, "repeats": len(measurements)}
                for key in METRICS:
                    values = [row[key] for row in measurements]
                    point[key] = statistics.mean(values)
                point["mean_latency_std_ms"] = statistics.stdev(row["mean_latency_ms"] for row in measurements) if len(measurements) > 1 else 0.0
                points.append(point)
                snapshot["points"].append(point)
    status = read_json(base / "status.json", {})
    if status.get("state") != "complete":
        incomplete.append("driver has not recorded state=complete")
    if status.get("state") == "complete":
        if status.get("graph_sha256") != graph_hash or status.get("models") != model_hashes:
            raise ValueError("driver's final graph/model fingerprints differ")
    if len(points) != 42:
        incomplete.append(f"observed {len(points)} of 42 operating points")
    metadata = {"dataset": dataset, "complete": not incomplete, "incomplete_reasons": incomplete,
                "expected_operating_points": 42, "expected_raw_rows": 42 * repeats,
                "observed_operating_points": len(points), "observed_raw_rows": sum(item["rows"] for item in snapshots),
                "query_ids_sha256": query_hash, "num_queries": query_count, "same_leaf_rotation_verified": same_rotation,
                "model_sha256": model_hashes, "graph_path": str(graph_path) if graph_path else None,
                "graph_sha256": graph_hash, "settings_sha256": settings_hash, "settings": settings, "curves": snapshots}
    if incomplete and points and not partial:
        raise ValueError("experiment is incomplete (use --partial for a snapshot): " + "; ".join(incomplete))
    return points, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_rows(points: list[dict], dataset: str) -> list[dict]:
    output = []
    for quant in QUANTS:
        for rerank in RERANKS:
            curve = [point for point in points if point["quantizer"] == quant and point["num_rerank"] == rerank]
            for target in (0.90, 0.95, 0.98, 0.99):
                candidates = [point for point in curve if point["recall_k_k"] >= target]
                best = min(candidates, key=lambda point: (point["mean_latency_ms"], point["L"])) if candidates else None
                output.append({"dataset": dataset, "quantizer": quant, "num_rerank": rerank, "target_recall": target,
                               "status": "reached" if best else "unreached_in_observed_grid",
                               "max_observed_recall": max((point["recall_k_k"] for point in curve), default=None),
                               "L": best["L"] if best else None, "observed_recall": best["recall_k_k"] if best else None,
                               "mean_latency_ms": best["mean_latency_ms"] if best else None,
                               "p95_latency_ms": best["p95_latency_ms"] if best else None})
    return output


def render(points: list[dict], metadata: dict, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    display = "FiQA" if metadata["dataset"] == "fiqa" else metadata["dataset"]
    footer = ("Same graph and leaf rotation; shared TQ8 reranker when enabled. Lines connect measured L settings; no extrapolation."
              if metadata["complete"] else "Partial snapshot: only written measurements are shown; curves without points are pending.")
    curves = {(item["quantizer"], item["num_rerank"]): item for item in metadata["curves"]}
    outputs = []

    def save(figure, name):
        for suffix in ("png", "pdf", "svg"):
            path = output / f"{name}.{suffix}"
            figure.savefig(path, dpi=200)
            outputs.append(str(path))
        plt.close(figure)

    for metric, name, xlabel in (("mean_latency_ms", "recall_vs_latency", "Mean query latency (ms, log scale)"),
                                 ("p95_latency_ms", "recall_vs_p95_latency", "p95 query latency (ms, log scale)")):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
        figure.suptitle(f"{display}: Vamana leaf quantization", fontsize=17, y=0.98)
        figure.text(0.5, 0.895, f"R=64 · build L=128 · alpha=1.0 · k=10 · {metadata['num_queries']:,} queries · one CPU thread", ha="center", fontsize=10)
        for axis, rerank in zip(axes, RERANKS):
            for quant, (label, color) in QUANTS.items():
                curve = curves[quant, rerank]
                if curve["status"] != "complete":
                    label += " (pending)" if not curve["points"] else " (in progress)"
                measurements = curve["points"]
                axis.plot([point[metric] for point in measurements], [100 * point["recall_k_k"] for point in measurements],
                          "o-", color=color, label=label, markersize=4.6, linewidth=1.9)
            axis.set(title="No reranking" if not rerank else f"TQ8 rerank: {rerank} candidates", xscale="log", xlabel=xlabel)
            axis.grid(which="major", alpha=0.25)
            axis.grid(which="minor", axis="x", alpha=0.1)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(loc="lower right", fontsize=9)
        axes[0].set_ylabel("Recall@10")
        recalls = [100 * point["recall_k_k"] for point in points]
        margin = max(1, (max(recalls) - min(recalls)) * 0.06)
        axes[0].set_ylim(max(0, min(recalls) - margin), min(100, max(recalls) + margin))
        figure.text(0.5, 0.037, footer, ha="center", fontsize=9, color="#555555")
        figure.subplots_adjust(top=0.79, bottom=0.19, left=0.06, right=0.99, wspace=0.12)
        save(figure, name)

    figure, axis = plt.subplots(figsize=(8.4, 6))
    for quant, (label, color) in QUANTS.items():
        for rerank, style, marker in zip(RERANKS, ("-", "--", ":"), ("o", "s", "^")):
            measurements = curves[quant, rerank]["points"]
            axis.plot([point["mean_latency_ms"] for point in measurements], [100 * point["recall_k_k"] for point in measurements],
                      color=color, linestyle=style, marker=marker, markersize=4.5, linewidth=1.8,
                      label=f"{label}; nr={rerank}")
    axis.set(title=f"{display}: leaf quantization and reranking", xscale="log",
             xlabel="Mean query latency (ms, log scale)", ylabel="Recall@10")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.grid(which="major", alpha=0.25)
    axis.grid(which="minor", axis="x", alpha=0.1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=9, ncol=2)
    figure.text(0.5, 0.025, "Partial snapshot" if not metadata["complete"] else "Same graph and rotation; every point is measured.",
                ha="center", fontsize=9, color="#555555")
    figure.subplots_adjust(bottom=0.14, left=0.1, right=0.98, top=0.92)
    save(figure, "recall_vs_latency_combined")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument("--results-root", type=Path, default=Path("experiments/vamana_ablations/quantization"))
    parser.add_argument("--partial", action="store_true", help="allow unfinished sweeps and label the snapshot")
    parser.add_argument("--no-plots", action="store_true", help="validate and write tables without importing Matplotlib")
    args = parser.parse_args()
    base = args.results_root.resolve() / args.dataset
    points, metadata = load_experiment(base, args.dataset, args.partial)
    output = base / "plots"
    output.mkdir(parents=True, exist_ok=True)
    metadata.update(snapshot_time_utc=datetime.now(timezone.utc).isoformat(),
                    aggregation="Arithmetic means across repeats at the same quantizer/rerank/L; no mixing between curves.",
                    target_selection="Fastest observed mean latency reaching the target; no interpolation.")
    if not points:
        metadata.update(status="pending", outputs=[])
        (output / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"No measurements yet for {args.dataset}; no plots generated.")
        return 0
    write_csv(output / "operating_points.csv", points)
    write_csv(output / "recall_targets.csv", target_rows(points, args.dataset))
    metadata["outputs"] = [] if args.no_plots else render(points, metadata, output)
    (output / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{'Complete' if metadata['complete'] else 'Partial'}: {len(points)}/42 operating points; outputs in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
