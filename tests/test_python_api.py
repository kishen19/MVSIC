"""Installed-extension ownership and numerical contract checks."""

import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

import mvsic


def dataset(n=24, vectors=3, d=32):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(n, vectors, d)).astype(np.float32)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return values


def raw_index(points, metric):
    params = mvsic.IndexParams.mvivf(max_leaf_size=1000)
    params.quantizer = mvsic.QuantizerType.None_
    params.quantize_centers = False
    index = mvsic.IndexMVIVF(points.dims, metric, params)
    index.build(points)
    return index


def test_exact_ragged_oracle(tmp_path):
    raw = dataset()
    values = raw.reshape(-1, raw.shape[-1])
    offsets = np.array([0, 2, 7, 12, 18, 25, 34, 42, 56, 72], dtype=np.uint64) * raw.shape[-1]
    ids = (np.arange(len(offsets) - 1, dtype=np.uint32) * 13) + 8
    points = mvsic.PointCloudSet(values, offsets, ids)
    index = raw_index(points, mvsic.Metric.IP)
    query = dataset(n=1, vectors=4)[0]
    distances = []
    for row in range(len(points)):
        distance = -(query @ points[row].T).max(axis=1).mean()
        distances.append(distance)
    order = np.lexsort((ids, distances))[:5]
    params = mvsic.SearchParams.mvivf(5, 1)
    found, scores = index.search(query, params)
    np.testing.assert_array_equal(found, ids[order])
    np.testing.assert_allclose(scores, np.array(distances)[order], atol=2e-6)
    path = str(tmp_path / "index.mvivf")
    index.save(path)
    reloaded = raw_index(points, mvsic.Metric.IP)
    reloaded.load(path, points)
    np.testing.assert_array_equal(reloaded.search(query, params)[0], found)


def test_l2_is_clustering_only():
    with pytest.raises(ValueError, match="MaxSim"):
        mvsic.IndexMVIVF(2, mvsic.Metric.L2)
    values = np.array([[[2., 0.]], [[4., 0.]]], dtype=np.float32)
    config = mvsic.MVClusteringConfig()
    config.niters = 1
    centers, assignments, _ = mvsic.cluster_point_clouds(
        mvsic.PointCloudSet(values), 1, 1, mvsic.Metric.L2, config)
    np.testing.assert_allclose(centers[0], [[3., 0.]], atol=1e-6)
    np.testing.assert_array_equal(assignments, [0, 0])


def test_numpy_ownership_copy_and_views():
    raw = dataset()
    ref = weakref.ref(raw)
    points = mvsic.PointCloudSet(raw)
    index = raw_index(points, mvsic.Metric.IP)
    query = raw[0].copy()
    view = points[0]
    assert not view.flags.writeable
    del points, raw
    gc.collect()
    assert ref() is not None
    assert index.search(query, mvsic.SearchParams.mvivf(1, 1))[0][0] == 0
    del index
    assert ref() is not None  # Returned cloud view still owns its buffer.
    del view
    gc.collect()
    assert ref() is None
    original = dataset()
    copied = mvsic.PointCloudSet(original, copy=True)
    saved = copied[0].copy()
    original[:] = 0
    np.testing.assert_array_equal(copied[0], saved)


def test_array_boundaries():
    raw = dataset()
    with pytest.raises(ValueError):
        mvsic.PointCloudSet(raw[:, :, ::2])
    with pytest.raises(ValueError):
        mvsic.PointCloudSet(raw.astype(np.float64))
    with pytest.raises(ValueError):
        mvsic.PointCloudSet(raw, ids=np.arange(len(raw), dtype=np.int64))
    # Zero-sized NumPy storage can have a huge leading dimension. Reject the
    # cloud count before attempting to allocate an offsets array of that size.
    with pytest.raises(ValueError):
        mvsic.PointCloudSet(np.empty((2**32, 0, 1), dtype=np.float32))
    points = mvsic.PointCloudSet(raw[:, :, ::2], copy=True)
    assert points.dims == raw.shape[-1] // 2
    index = raw_index(points, mvsic.Metric.IP)
    query = points[0].copy()
    query[0, 0] = np.nan
    with pytest.raises(ValueError):
        index.search(query, mvsic.SearchParams.mvivf(1, 1))


@pytest.mark.parametrize("codec", ["None_", "TQ", "OneBitTQ", "EightBitTQ"])
def test_batch_rerank_and_native_benchmark(codec):
    raw = dataset(n=40, d=128)
    points = mvsic.PointCloudSet(raw)
    params = mvsic.IndexParams.mvivf(k_per_level=4, max_leaf_size=12, niters=2, s=3)
    params.quantizer = getattr(mvsic.QuantizerType, codec)
    index = mvsic.IndexMVIVF(128, mvsic.Metric.IP, params)
    index.build(points)
    queries = mvsic.PointCloudSet(raw[:8])
    search = mvsic.SearchParams.mvivf(5, 100, num_rerank=20, tq8_rerank=True)
    batch, distances = index.search_all(queries, search)
    for i in range(len(queries)):
        ids, scores = index.search(queries[i], search)
        np.testing.assert_array_equal(batch[i], ids)
        np.testing.assert_allclose(distances[i], scores, atol=2e-5)
    native = mvsic.benchmark_native(index, queries, search, "latency_single", 1, 2)
    assert len(native["seconds"]) == 2
    assert np.asarray(native["query_seconds"]).shape == (2, 8)
    np.testing.assert_array_equal(native["ids"], batch)


def test_reject_unavailable_backend():
    caps = mvsic.capabilities()
    for backend in ("vnni", "amx"):
        if backend == "vnni" and caps["usable"]["vnni"]:
            continue
        code = """import mvsic, numpy as np
p=mvsic.IndexParams.mvivf()
i=mvsic.IndexMVIVF(32,mvsic.Metric.IP,p)
x=np.ones((2,1,32),dtype=np.float32)/np.sqrt(np.float32(32))
try:
    i.build(mvsic.PointCloudSet(x))
except ValueError:
    pass
else:
    raise AssertionError('unavailable ISA accepted')
"""
        env = dict(os.environ, MVSIC_CPU_ISA=backend, PARLAY_NUM_THREADS="1")
        subprocess.run([sys.executable, "-c", code], env=env, check=True)
