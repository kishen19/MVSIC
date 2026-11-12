"""Multi-vector similarity search. Native code loads when an index is requested."""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["Metric", "ClusteringQuantizer", "QuantizerType", "QueryCompression",
           "PointCloudSet", "IndexParams", "SearchParams", "MVClusteringConfig", "Index", "IndexMVIVF",
           "capabilities", "cluster_point_clouds", "benchmark_native"]


def __getattr__(name):
    if name in __all__:
        native = import_module("._mvsic", __name__)
        value = getattr(native, "benchmark" if name == "benchmark_native" else name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
