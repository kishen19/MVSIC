"""Run after installing the wheel, from an environment with NumPy and pytest.

Every CLI subprocess runs outside the checkout with PYTHONPATH removed. Optional
plot dependencies are tested either by rendering or by checking the install hint.
"""

import importlib.util
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]


def write_pcs(path, values):
    clouds, vectors, dimensions = values.shape
    offsets = np.arange(clouds + 1, dtype=np.uint64) * vectors * dimensions
    with path.open("wb") as stream:
        np.array([dimensions, clouds, clouds * vectors], dtype="<u8").tofile(stream)
        values.astype("<f4").tofile(stream)
        np.array([clouds + 1], dtype="<u8").tofile(stream)
        offsets.astype("<u8").tofile(stream)


def write_gt(path, points, queries):
    distances = np.array([[-(query @ point.T).max(axis=1).mean() for point in points]
                          for query in queries], dtype=np.float32)
    order = np.argsort(distances, axis=1, kind="stable")
    records = np.empty(order.shape, dtype=[("distance", "<f4"), ("id", "<u4")])
    records["id"] = order
    records["distance"] = np.take_along_axis(distances, order, axis=1)
    with path.open("wb") as stream:
        np.array([len(points)], dtype="<i4").tofile(stream)
        records.tofile(stream)


def test_installed_cli_complete_flow(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PARLAY_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    inspect = subprocess.run(
        [sys.executable, "-c", "import mvsic; from importlib.resources import files; "
         "print(mvsic.__file__); print(files('mvsic.benchmark').joinpath('configs/mvivf.build.yaml').is_file())"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    lines = inspect.stdout.splitlines()
    assert not Path(lines[0]).resolve().is_relative_to(REPOSITORY), "run this test against an installed wheel"
    assert lines[1] == "True", "installed example config is missing"
    console = Path(sys.executable).parent / "mvsic"
    assert console.is_file(), "console entry point is missing"

    rng = np.random.default_rng(17)
    points = rng.normal(size=(12, 2, 32)).astype(np.float32)
    queries = rng.normal(size=(4, 3, 32)).astype(np.float32)
    points /= np.linalg.norm(points, axis=2, keepdims=True)
    queries /= np.linalg.norm(queries, axis=2, keepdims=True)
    write_pcs(tmp_path / "points.pcs", points)
    write_pcs(tmp_path / "queries.pcs", queries)
    write_gt(tmp_path / "truth.gt", points, queries)
    config = {
        "version": 1, "name": "installed smoke",
        "dataset": {"points": "points.pcs", "queries": "queries.pcs", "ground_truth": "truth.gt"},
        "index": {"metric": "ip", "quantizer": "None", "quantize_centers": False,
                  "params": {"max_leaf_size": 64}},
        "build": {"output": "tiny.idx", "threads": 1},
        "search": {"k": 3, "nprobes": [1, 2], "num_rerank": 0},
        "measurement": {"threads": 1, "repeats": 2, "warmup_queries": 2},
        "output": {"directory": "runs"},
    }
    filename = tmp_path / "plan.json"
    filename.write_text(json.dumps(config))

    def run(action, success=True):
        result = subprocess.run([str(console), "benchmark", action, "--config", str(filename)],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        if success:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0, result.stdout + result.stderr
        return result

    validated = json.loads(run("validate").stdout)
    assert validated["operating_points"] == 2
    assert validated["resolved"]["dataset"]["points"] == str(tmp_path / "points.pcs")
    build_directory = Path(run("build").stdout.strip().splitlines()[-1])
    build = json.loads((build_directory / "build.json").read_text())
    assert build["build_seconds"] > 0
    assert build["runtime"]["native_libraries"]["resolved"], "compiled implementation DSOs need provenance"
    search_directory = Path(run("search").stdout.strip().splitlines()[-1])
    records = sorted(search_directory.glob("search-*.json"))
    # Job/process metadata have different names; per-point JSONs are SHA-named.
    records = [path for path in records if len(path.stem.removeprefix("search-")) == 64]
    assert len(records) == 2
    for path in records:
        record = json.loads(path.read_text())
        assert record["status"] == "complete"
        assert record["recall_at_k"] == 1.0
        assert len(record["seconds"]) == 2
        assert record["latency_seconds"]["p99"] > 0
        assert record["warmup_seconds"] >= record["first_call_seconds"] > 0
        assert record["first_call_scope"] == "single_query"
        assert len(record["python_seconds"]) == 2
        with np.load(record["outputs"][0]["path"]) as arrays:
            assert arrays["ids"].shape == (4, 3)
            assert arrays["query_seconds"].shape == (2, 4)
    timestamps = [path.stat().st_mtime_ns for path in records]
    run("search")
    assert timestamps == [path.stat().st_mtime_ns for path in records], "matching complete results should resume"

    if importlib.util.find_spec("matplotlib"):
        plotted = run("plot")
        assert Path(plotted.stdout.strip().splitlines()[-1]).read_bytes().startswith(b"\x89PNG")
    else:
        assert 'mvsic[plot]' in run("plot", success=False).stderr

    # Changing search policy reuses the same build, then exercises search_all.
    config["measurement"]["mode"] = "throughput_batch"
    filename.write_text(json.dumps(config))
    batch_directory = Path(run("search").stdout.strip().splitlines()[-1])
    batch_record = next(path for path in batch_directory.glob("search-*.json")
                        if len(path.stem.removeprefix("search-")) == 64)
    record = json.loads(batch_record.read_text())
    assert record["latency_seconds"] is None
    assert record["first_call_scope"] == "whole_batch"
    assert record["warmup_seconds"] >= record["first_call_seconds"] > 0
    assert record["recall_at_k"] == 1.0

    # A changed build parameter must fail before an incompatible skeleton is used.
    config["index"]["params"]["seed"] = 1
    filename.write_text(json.dumps(config))
    failed = run("search", success=False)
    assert "search failed" in failed.stderr


def test_installed_original_yaml_workflow(tmp_path):
    """Install, create existing YAML, load/build/search/plot outside the checkout."""
    import yaml
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PARLAY_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    console = Path(sys.executable).parent / "mvsic"
    rng = np.random.default_rng(120)
    points = rng.normal(size=(12, 2, 32)).astype(np.float32)
    queries = rng.normal(size=(4, 3, 32)).astype(np.float32)
    points /= np.linalg.norm(points, axis=2, keepdims=True)
    queries /= np.linalg.norm(queries, axis=2, keepdims=True)
    data = tmp_path / "data/tiny"
    data.mkdir(parents=True)
    write_pcs(data / "tiny_points.pcs", points)
    write_pcs(data / "tiny_queries.pcs", queries)
    write_gt(data / "tiny_chamfer_neighbors.gt", points, queries)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    build_file = config_dir / "build.yaml"
    # Paths are CWD-relative, not relative to configs/.
    build_file.write_text("""name: Existing build schema
datasets:
  - {name: tiny, path: data/tiny, index_dir: indices/tiny, is_mmap: true}
indices:
  - name: mvivf
    metric: ip
    build_configs:
      - name: tree
        params: &settings {max_leaf_size: 64, niters: 2, seed: 0, inner_metric: l2}
      - name: second_tree
        params: {<<: *settings, seed: 7}
""")
    result = subprocess.run([sys.executable, "-c",
                            "from mvsic.benchmark import load_config, run; "
                            "p=load_config('configs/build.yaml'); "
                            "assert p.validate()['jobs']==2; run('build',p,build_threads=1)"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("tree", "second_tree"):
        directory = tmp_path / f"indices/tiny/mvivf/{name}"
        assert (directory / "index.bin").is_file()
        stats = json.loads((directory / "build_stats.json").read_text())
        assert stats["build_time_sec"] > 0
        assert stats["build_params"]["inner_metric"] == "l2"
        assert (directory / "index_params.json").is_file()
    artifact = tmp_path / "indices/tiny/mvivf/tree/index.bin"
    built_time = artifact.stat().st_mtime_ns
    search_file = config_dir / "search.yaml"
    search = {
        "name": "Existing search schema",
        "datasets": [{"name": "tiny", "path": "data/tiny", "index_dir": "indices/tiny",
                      "results_dir": "results/tiny", "is_mmap": True}],
        "indices": [{"name": "mvivf", "metric": "ip", "builds": [{
            "build_name": "tree", "build_params": {"max_leaf_size": 64, "niters": 2, "seed": 0, "inner_metric": "l2"},
            "variants": [{"name": "", "compress": False, "quantizer": "None"},
                         {"name": "tq8", "compress": True, "quantizer": "EightBitTQ"}],
            "search_configs": [{"name": "k=3", "params": {
                "k": [3], "nprobes": [1, 2], "tq8_rerank": [False],
                "config": [{"name": "nr0", "num_rerank": 0},
                           {"name": "nr12", "num_rerank": 12, "nprobes": [1, 4]}]}}]}]}]}
    search_file.write_text(yaml.safe_dump(search, sort_keys=False))

    def cli(action, filename, *options, success=True):
        result = subprocess.run([str(console), "benchmark", action, "--config", str(filename), *options],
                                cwd=tmp_path, env=env, text=True, capture_output=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    assert json.loads(cli("validate", search_file, "--latency").stdout)["operating_points"] == 8
    for mode in ("latency", "multi_latency", "batch"):
        cli("search", search_file, "--mode", mode, "--num_threads", "1", "--query_subsample", "0", "--QPS_thresh", "0")
        for variant in ("", "tq8"):
            directory = tmp_path / "results/tiny/mvivf/tree" / variant / "k=3"
            for label in ("nr0", "nr12"):
                with (directory / f"{mode}_results_{label}.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                assert rows and all(int(row["num_rerank"]) == int(label[2:]) for row in rows)
                qps_field = "QPS_seq" if mode == "latency" else "QPS_par"
                assert float(rows[0][qps_field]) > 0
                assert rows[0]["QPS_par" if mode == "latency" else "QPS_seq"] == ""
                if label == "nr12":
                    assert float(rows[0]["recall_k_k"]) == 1
                    assert len(rows) == 1, "the original recall=1 stop rule must apply"
    assert artifact.stat().st_mtime_ns == built_time, "serving variants must reuse one structural artifact"
    raw_records = list((tmp_path / "results/tiny").rglob("search-*.json"))
    raw_records = [path for path in raw_records if len(path.stem.removeprefix("search-")) == 64]
    timestamps = {str(path): path.stat().st_mtime_ns for path in raw_records}
    cli("search", search_file, "--mode", "latency", "--num_threads", "1", "--query_subsample", "0", "--QPS_thresh", "0")
    assert timestamps == {str(path): path.stat().st_mtime_ns for path in raw_records}
    assert any(path.suffix == ".npz" for path in (tmp_path / "results/tiny").rglob("*.npz"))

    # The older combined config also runs directly. Its default measurement
    # writes both sequential-query and batch QPS to an unprefixed CSV.
    combined = {"name": "combined", "datasets": [{"name": "tiny", "path": "data/tiny", "results": "combined/tiny"}],
                "indices": [{"name": "mvivf", "build_configs": [{"name": "tree",
                    "params": {"max_leaf_size": 64, "pq_method": 0},
                    "search_configs": [{"name": "k=3", "params": {"k": [3], "nprobes": [1, 2],
                                        "config": [{"name": "", "num_rerank": 0}]}}]}]}]}
    combined_file = config_dir / "combined.yaml"
    combined_file.write_text(yaml.safe_dump(combined))
    cli("build", combined_file, "--build-threads", "1")
    cli("search", combined_file, "--query_subsample", "0")
    with (tmp_path / "combined/tiny/mvivf/tree/k=3/results.csv").open() as stream:
        combined_row = next(csv.DictReader(stream))
    assert float(combined_row["QPS_seq"]) > 0
    assert float(combined_row["QPS_par"]) > 0

    plot_file = config_dir / "plot.yaml"
    plot_file.write_text("""name: installed_plots
datasets:
  - {name: tiny, results: results/tiny}
plots:
  - name: variants
    k: 3
    results:
      - [mvivf, tree, '', tq8]
""")
    cli("validate", plot_file)
    if importlib.util.find_spec("matplotlib"):
        cli("plot", plot_file, "--affix", "smoke", "--format", "png")
        assert list((tmp_path / "results/installed_plots").glob("*.png"))
        assert len(list((tmp_path / "results/installed_plots").glob("*.csv"))) == 3
    else:
        assert "mvsic[plot]" in cli("plot", plot_file, success=False).stderr

    search["indices"][0]["builds"][0]["build_params"]["seed"] = 99
    search_file.write_text(yaml.safe_dump(search))
    assert "build parameters" in cli("search", search_file, "--latency", success=False).stderr
