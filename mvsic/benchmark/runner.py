"""Isolated benchmark jobs and content-verified result storage."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

from .config import digest


RESULT_VERSION = 1
THREAD_POLICY_VERSION = 1
LIBRARY_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
)
THREAD_ENVIRONMENT = ("PARLAY_NUM_THREADS", *LIBRARY_THREAD_ENVIRONMENT)
KERNEL_ENVIRONMENT = (
    "MVSIC_CPU_ISA", "MVIVF_SUBBATCH", "TSC_GHZ", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC", "OMP_WAIT_POLICY", "OMP_MAX_ACTIVE_LEVELS",
    "KMP_AFFINITY", "KMP_BLOCKTIME", "GOMP_CPU_AFFINITY", "MKL_DYNAMIC", "OPENBLAS_CORETYPE",
    "GLIBC_TUNABLES", "MALLOC_CONF", "MALLOC_ARENA_MAX",
)


def worker_environment(action, config, environment):
    """Resolve limits before importing any native library in the child process."""
    env = dict(environment)
    provided = {name: env.get(name) for name in THREAD_ENVIRONMENT}
    execution = config["build"] if action == "build" else config["measurement"]
    env["PARLAY_NUM_THREADS"] = str(execution["threads"])
    single = action == "search" and execution["mode"] == "latency_single"
    for name in LIBRARY_THREAD_ENVIRONMENT:
        if single:
            env[name] = "1"
        else:
            env.setdefault(name, "1")
    return env, {"policy_version": THREAD_POLICY_VERSION, "provided": provided,
                 "effective": {name: env[name] for name in THREAD_ENVIRONMENT}}


def execution_runtime(runtime, config, environment):
    # Both limits are retained: search concurrency must not change a build's identity.
    return {**runtime, "thread_environment": {
        "policy_version": THREAD_POLICY_VERSION,
        **{action: worker_environment(action, config, environment)[1]["effective"]
           for action in ("build", "search")}}}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_identity(filename):
    path = Path(filename).resolve(strict=True)
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"input changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": sha.hexdigest()}


def linked_libraries(filename):
    """Resolve actual MVSIC DSOs, including development builds with build RPATHs."""
    result = subprocess.run(["ldd", str(filename)], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"cannot inspect native shared libraries: {result.stderr or result.stdout}")
    libraries = {}
    for line in result.stdout.splitlines():
        if "libmvsic" not in line:
            continue
        match = re.match(r"\s*(libmvsic\S+)\s+=>\s+(\S+)\s+\(", line)
        if match is None:
            raise RuntimeError(f"unresolved native shared library: {line.strip()}")
        libraries[match[1]] = file_identity(match[2])
    return libraries


def runtime_identity():
    import mvsic
    spec = importlib.util.find_spec("mvsic._mvsic")
    if spec is None or spec.origin is None:
        raise RuntimeError("the native extension is not installed; install mvsic before build/search")
    bundled = {}
    for directory in (Path(spec.origin).parent / ".libs", Path(spec.origin).parent):
        for path in sorted(directory.glob("libmvsic*.so*")):
            if path.is_file():
                bundled[path.name] = file_identity(path)
    libraries = {"bundled": bundled, "resolved": linked_libraries(spec.origin)}
    sources = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cpu = {}
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().split("\n\n", 1)[0].splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("model name", "flags", "vendor_id"):
                cpu[key.strip()] = value.strip()
    return {"package_version": mvsic.__version__, "native": file_identity(spec.origin),
            "native_libraries": libraries, "numpy_version": importlib.metadata.version("numpy"),
            "environment": {name: os.environ.get(name) for name in KERNEL_ENVIRONMENT},
            "harness_sha256": digest(sources), "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


@contextmanager
def writer_lock(path):
    """Advisory OS lock: process death releases it, without stale-lock guessing."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another benchmark is writing {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def build_identity(config, inputs, runtime):
    # Search sweeps and output paths do not change the built structure/representation.
    return digest({"version": RESULT_VERSION, "index": config["index"], "points": inputs["points"],
                   "execution": {key: config["build"][key] for key in ("threads", "cpu")},
                   "native": runtime["native"], "native_libraries": runtime.get("native_libraries", {}),
                   "environment": runtime.get("environment", {}),
                   "thread_environment": {"policy_version": THREAD_POLICY_VERSION,
                                          "limits": runtime.get("thread_environment", {}).get("build", {})},
                   "numpy_version": runtime.get("numpy_version"), "package_version": runtime["package_version"]})


def completed(path, fingerprint, artifact=None):
    path = Path(path)
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except (ValueError, OSError):
        return False
    if result.get("status") != "complete" or result.get("fingerprint") != fingerprint:
        return False
    outputs = result.get("outputs", [])
    if artifact is None and not outputs:
        return False
    if artifact is not None:
        outputs = [result.get("artifact", {})]
        if not outputs[0] or str(Path(artifact).resolve()) != outputs[0].get("path"):
            return False
    for expected in outputs:
        try:
            if file_identity(expected["path"]) != expected:
                return False
        except (OSError, KeyError):
            return False
    return True


def _worker(action, config, context, directory, force, **extra):
    job = {"action": action, "config": config, "context": context, "directory": str(directory), "force": force}
    job.update(extra)
    job_path = directory / f"{action}-job.json"
    atomic_json(job_path, job)
    env, thread_environment = worker_environment(action, config, os.environ)
    log_path = directory / f"{action}.log"
    with log_path.open("w") as log:
        result = subprocess.run([sys.executable, "-m", "mvsic.benchmark.worker", str(job_path)],
                                env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    atomic_json(directory / f"{action}-process.json",
                {"status": "complete" if result.returncode == 0 else "failed",
                 "returncode": result.returncode, "log": str(log_path),
                 "thread_environment": thread_environment})
    if result.returncode:
        raise RuntimeError(f"{action} failed (exit {result.returncode}); see {log_path}")


def run(action, config, force=False):
    if action == "plot":
        from .plotting import plot
        return plot(config)
    value = config.resolved
    dataset = value["dataset"]
    if action == "search" and "queries" not in dataset:
        raise ValueError("search requires dataset.queries")
    inputs = {key: file_identity(dataset[key]) for key in ("points", "queries", "ground_truth")
              if key in dataset and (key == "points" or action == "search")}
    runtime = execution_runtime(runtime_identity(), value, os.environ)
    context = {"schema_version": RESULT_VERSION, "inputs": inputs, "runtime": runtime,
               "config_fingerprint": config.fingerprint}
    context["build_fingerprint"] = build_identity(value, inputs, runtime)
    identity = digest({"config": value, "inputs": inputs, "runtime": runtime})
    directory = Path(value["output"]["directory"]) / identity
    directory.mkdir(parents=True, exist_ok=True)
    with writer_lock(directory / ".lock"):
        atomic_json(directory / "manifest.json", context)
        atomic_json(directory / "resolved_config.json", value)
        if action == "build":
            with writer_lock(Path(value["build"]["output"] + ".lock")):
                _worker(action, value, context, directory, force)
        else:
            # Also hold the artifact lock while loading/measuring so a concurrent build
            # cannot replace the skeleton after provenance validation.
            with writer_lock(Path(value["build"]["output"] + ".lock")):
                _worker(action, value, context, directory, force)
    print(directory)
    return directory
