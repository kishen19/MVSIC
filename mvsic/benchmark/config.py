"""Resolve and validate a benchmark plan before importing native code."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path


BUILD_DEFAULTS = {
    "k_per_level": 0, "max_leaf_size": 500, "max_depth": 0,
    "compress_input": False, "verbose": 0, "niters": 5,
    "max_point_clouds_per_cluster": 100,
    "max_points_per_centroid_inner_kmeans": 20, "init": "Random", "seed": 0,
    "use_weighted_inner_kmeans": True, "s": 0,
    "build_with_8btq": False, "assignment_quantizer": "None",
    "inner_metric": "same", "inner_init": "default", "inner_backend": "legacy",
}
SEARCH_DEFAULTS = {
    "k": 10, "nprobes": [1, 2, 4, 8, 16, 32], "num_rerank": [0, 20, 40],
    "tq8_rerank": True, "query_compression": "None",
    "query_compression_threshold": 0.7, "compress_rerank": False,
    "query_alignment": 0, "query_alignment_strict": False, "root_m2m": True,
}
BUILD_PARAMS = set(BUILD_DEFAULTS)
SEARCH_PARAMS = set(SEARCH_DEFAULTS)


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _keys(value, allowed, where):
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"unknown {where} fields: {', '.join(map(str, sorted(extra, key=str)))}")


def _integer(value, name, zero=False, maximum=2**32 - 1):
    if type(value) is not int or not (0 if zero else 1) <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{0 if zero else 1}, {maximum}]")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _string(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config field: {key}")
        result[key] = value
    return result


def read_config(path):
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(), object_pairs_hook=_unique_object)
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('YAML plans require pip install "mvsic[benchmark]"; JSON needs no YAML package') from error

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        # YAML merge keys intentionally allow explicit fields to override anchors.
        # Reject duplicate explicit fields, not those inherited from a merge.
        _unique_object((loader.construct_object(k, deep=deep), None)
                       for k, _ in node.value if k.tag != 'tag:yaml.org,2002:merge')
        loader.flatten_mapping(node)
        return {loader.construct_object(k, deep=deep): loader.construct_object(v, deep=deep)
                for k, v in node.value}

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(path.read_text(), Loader=UniqueLoader)


@dataclass(frozen=True)
class BenchmarkConfig:
    source: Path
    resolved: dict

    @property
    def fingerprint(self):
        return digest(self.resolved)

    def points(self):
        search = self.resolved["search"]
        axes = {key: value if isinstance(value, list) else [value] for key, value in search.items()}
        names = list(axes)
        return [dict(zip(names, values)) for values in itertools.product(*(axes[name] for name in names))]


def load_config(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    return resolve_config(read_config(path), path)


def resolve_config(raw, source):
    path = Path(source).resolve()
    raw = _object(raw, "config")
    if "datasets" in raw or "indices" in raw:
        raise ValueError("dataset workflows are accepted directly by 'mvsic benchmark --help'; this helper resolves one internal job")
    _keys(raw, {"version", "name", "dataset", "index", "build", "search", "measurement", "output"}, "config")
    if type(raw.get("version")) is not int or raw["version"] != 1:
        raise ValueError("benchmark config version must be 1")
    value = json.loads(json.dumps(raw, allow_nan=False))
    value.setdefault("name", path.stem)
    _string(value["name"], "name")

    def resolve(name):
        item = Path(_string(name, "path")).expanduser()
        return str((item if item.is_absolute() else path.parent / item).resolve())

    dataset = _object(value.get("dataset"), "dataset")
    _keys(dataset, {"name", "points", "queries", "ground_truth", "mmap"}, "dataset")
    if "points" not in dataset:
        raise ValueError("dataset.points is required")
    for key in ("points", "queries", "ground_truth"):
        if key in dataset:
            dataset[key] = resolve(dataset[key])
    dataset.setdefault("name", Path(dataset["points"]).stem)
    _string(dataset["name"], "dataset.name")
    dataset.setdefault("mmap", True)
    _boolean(dataset["mmap"], "dataset.mmap")
    index = _object(value.setdefault("index", {}), "index")
    _keys(index, {"name", "metric", "quantizer", "quantize_centers", "params"}, "index")
    index.setdefault("name", "mvivf")
    index.setdefault("metric", "ip")
    index.setdefault("quantizer", "OneBitTQ")
    index.setdefault("quantize_centers", True)
    _choice(index["name"], ("mvivf",), "index.name")
    if index["metric"] != "ip":
        raise ValueError("MVIVF retrieval supports only ip (MaxSim); use inner_metric for L2 clustering")
    _boolean(index["quantize_centers"], "index.quantize_centers")
    aliases = {"float": "None", "TQ4": "TQ", "TurboQuant": "TQ", "TQ1": "OneBitTQ", "TQ8": "EightBitTQ"}
    _string(index["quantizer"], "index.quantizer")
    index["quantizer"] = aliases.get(index["quantizer"], index["quantizer"])
    _choice(index["quantizer"], ("None", "TQ", "OneBitTQ", "EightBitTQ"), "index.quantizer")
    params = _object(index.setdefault("params", {}), "index.params")
    _keys(params, BUILD_PARAMS, "index.params")
    for key, default in BUILD_DEFAULTS.items():
        params.setdefault(key, default)
    for key in ("k_per_level", "max_depth", "verbose", "niters", "s", "seed", "max_point_clouds_per_cluster"):
        _integer(params[key], f"index.params.{key}", zero=True)
    for key in ("max_leaf_size", "max_points_per_centroid_inner_kmeans"):
        _integer(params[key], f"index.params.{key}")
    for key in ("compress_input", "use_weighted_inner_kmeans", "build_with_8btq"):
        _boolean(params[key], f"index.params.{key}")
    _choice(params["init"], ("Random",), "index.params.init")
    _choice(params["inner_metric"], ("same", "l2", "ip"), "index.params.inner_metric")
    _choice(params["inner_init"], ("default", "uniform", "kmeans++", "prefix_doubling"), "index.params.inner_init")
    _choice(params["inner_backend"], ("legacy", "adaptive"), "index.params.inner_backend")
    if params["assignment_quantizer"] == "float":
        params["assignment_quantizer"] = "None"
    _choice(params["assignment_quantizer"], ("None", "TQ8", "TQ4", "TQ1"), "assignment_quantizer")
    if params["build_with_8btq"]:
        if params["assignment_quantizer"] not in ("None", "TQ8"):
            raise ValueError("build_with_8btq conflicts with assignment_quantizer")
        params["assignment_quantizer"] = "TQ8"
        params["build_with_8btq"] = False  # Canonicalize the old alias.

    build = _object(value.setdefault("build", {}), "build")
    _keys(build, {"output", "threads", "cpu"}, "build")
    build["output"] = resolve(build.get("output", "results/mvivf.idx"))
    build.setdefault("threads", 1)
    build.setdefault("cpu", None)
    _integer(build["threads"], "build.threads")
    if build["cpu"] is not None:
        _integer(build["cpu"], "build.cpu", zero=True)
        if build["threads"] != 1:
            raise ValueError("CPU pinning requires build.threads=1")
    search = _object(value.setdefault("search", {}), "search")
    _keys(search, SEARCH_PARAMS, "search")
    for key, default in SEARCH_DEFAULTS.items():
        search.setdefault(key, default)
    combinations = 1
    for key, axis in search.items():
        items = axis if isinstance(axis, list) else [axis]
        if not items:
            raise ValueError(f"search.{key} cannot be empty")
        combinations *= len(items)
        if combinations > 100000:
            raise ValueError("search grid exceeds 100000 operating points")
        for item in items:
            name = f"search.{key}"
            if key in ("k", "nprobes", "num_rerank", "query_alignment"):
                _integer(item, name, zero=key in ("num_rerank", "query_alignment"))
            elif key == "query_compression":
                _choice(item, ("None", "Carve", "Wards"), name)
            elif key == "query_compression_threshold":
                if type(item) not in (int, float) or not math.isfinite(item):
                    raise ValueError(f"{name} must be finite")
            else:
                _boolean(item, name)

    measurement = _object(value.setdefault("measurement", {}), "measurement")
    defaults = {"mode": "latency_single", "threads": 1, "cpu": None, "warmup_queries": 10,
                "repeats": 3, "query_limit": 0, "query_seed": 42, "python_timing": True}
    _keys(measurement, defaults, "measurement")
    for key, default in defaults.items():
        measurement.setdefault(key, default)
    _choice(measurement["mode"], ("latency_single", "latency_parallel", "throughput_batch"), "measurement.mode")
    for key in ("threads", "warmup_queries", "repeats", "query_limit", "query_seed"):
        _integer(measurement[key], f"measurement.{key}", zero=key in ("warmup_queries", "query_limit", "query_seed"),
                 maximum=100000 if key in ("repeats", "warmup_queries") else 2**32 - 1)
    _boolean(measurement["python_timing"], "measurement.python_timing")
    if measurement["mode"] == "latency_single" and measurement["threads"] != 1:
        raise ValueError("latency_single requires measurement.threads=1")
    if measurement["cpu"] is not None:
        _integer(measurement["cpu"], "measurement.cpu", zero=True)
        if measurement["threads"] != 1:
            raise ValueError("CPU pinning requires measurement.threads=1")
    output = _object(value.setdefault("output", {}), "output")
    _keys(output, {"directory"}, "output")
    output["directory"] = resolve(output.get("directory", "results/benchmark"))
    if build["output"] in {dataset.get(key) for key in ("points", "queries", "ground_truth")}:
        raise ValueError("build.output must differ from dataset input paths")
    result = BenchmarkConfig(path, value)
    for point in result.points():
        if point["num_rerank"] and point["num_rerank"] < point["k"]:
            raise ValueError("num_rerank must be 0 or at least k")
        if point["query_compression"] == "Wards" and point["query_compression_threshold"] < 0:
            raise ValueError("Wards compression threshold must be nonnegative")
        if point["query_alignment_strict"] and not point["query_alignment"]:
            raise ValueError("query_alignment_strict requires positive query_alignment")
    return result
