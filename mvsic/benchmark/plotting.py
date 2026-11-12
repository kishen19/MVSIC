"""Plots are derived from complete, full-precision result records."""

import json
from pathlib import Path

from .config import digest


def plot(config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('plotting requires pip install "mvsic[plot]"') from error
    root = Path(config.resolved["output"]["directory"])
    groups = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_fingerprint") != config.fingerprint:
            continue
        for path in sorted(manifest_path.parent.glob("search-*.json")):
            record = json.loads(path.read_text())
            if record.get("status") != "complete" or record.get("recall_at_k") is None:
                continue
            point = {key: value for key, value in record["params"].items() if key != "nprobes"}
            identity = {"params": point, "measurement": record["measurement"],
                        "runtime": manifest["runtime"]}
            key = digest(identity)
            label = (f'k={point["k"]}, rerank={point["num_rerank"]}, '
                     f'{point["query_compression"]}, {record["measurement"]["mode"]}, '
                     f'{record["measurement"]["threads"]} threads, {key[:8]}')
            groups.setdefault(key, (label, []))[1].append((record["recall_at_k"], record["qps"]))
    if not groups:
        raise ValueError("no complete search results with ground truth match this config")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for key in sorted(groups):
        label, values = groups[key]
        values.sort()
        axis.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=label)
    axis.set(xlabel="Recall@k (boundary ties)", ylabel="Queries / second", title=config.resolved["name"])
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    output = root / "plots" / f"{config.fingerprint[:16]}-recall-qps.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)
    return output
