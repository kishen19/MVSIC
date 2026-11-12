"""Controlled MVIVF comparison; run only after the coordination marker is clear.

This harness consumes explicitly built native_bench executables. It never writes
inside the original repo and copies executables to content-addressed paths before
running them. See README.md for compilation plans and measurement scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time


DEFAULT_MARKER = "/home/kishen/MVSIC/emdivf_experiments/artifacts/EMDIVF_BENCHMARK_ACTIVE.json"
RESULT_ROOT = Path("/ssd2/kishen/mvsic_new")
SEARCH_SCOPES = ("shared_original", "independent_build")


def process_snapshot():
    """Read Linux process CPU counters; only a process exiting is harmless."""
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        end = raw.rindex(")")
        fields = raw[end + 2:].split()
        result[int(entry.name)] = {
            "name": raw[raw.index("(") + 1:end], "ppid": int(fields[1]),
            "ticks": int(fields[11]) + int(fields[12]), "start": int(fields[19]),
        }
    if os.getpid() not in result:
        raise RuntimeError("host snapshot omitted the experiment controller")
    return result


def expand_smt_cpus(cpus, siblings=None):
    """Reserve each configured CPU's entire physical core."""
    reserved = set()
    for cpu in cpus:
        if siblings is not None:
            group = set(siblings[cpu])
        else:
            value = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
            group = set()
            for part in value.split(","):
                bounds = part.split("-")
                if len(bounds) == 1:
                    group.add(int(bounds[0]))
                elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
                    group.update(range(int(bounds[0]), int(bounds[1]) + 1))
                else:
                    raise ValueError("invalid SMT sibling CPU list")
        if cpu not in group:
            raise ValueError("CPU is absent from its SMT sibling list")
        reserved.update(group)
    if not reserved:
        raise ValueError("shared-host mode requires explicit worker CPU affinities")
    return reserved


def process_thread_affinities(pid):
    """Read every thread's allowed CPUs, failing closed on an unstable task set."""
    directory = Path(f"/proc/{pid}/task")
    tids = {int(entry.name) for entry in directory.iterdir() if entry.name.isdigit()}
    if not tids:
        raise RuntimeError(f"no threads found for process {pid}")
    result = {tid: set(os.sched_getaffinity(tid)) for tid in sorted(tids)}
    after = {int(entry.name) for entry in directory.iterdir() if entry.name.isdigit()}
    if after != tids or any(not mask for mask in result.values()):
        raise RuntimeError(f"thread affinities changed while inspecting process {pid}")
    return result


def shared_host_processes(active, reserved_cpus, affinity_reader):
    """Keep only potentially overlapping CPU work in the contention pool."""
    overlapping = []
    for row in active:
        masks = affinity_reader(row["pid"])
        if not masks or any(not mask for mask in masks.values()):
            raise RuntimeError("cannot establish every thread's CPU affinity")
        allowed = set().union(*masks.values())
        row["thread_affinities"] = {str(tid): sorted(mask) for tid, mask in masks.items()}
        row["allowed_cpus"] = sorted(allowed)
        row["overlaps_reserved"] = bool(allowed & reserved_cpus)
        if row["overlaps_reserved"]:
            overlapping.append(row)
    return overlapping


class HostLoadState:
    """Pure CPU-delta accounting, independent of /proc, clocks and signals."""

    def __init__(self, owner, ticks_per_second, *, reserved_cpus=None, affinity_reader=None):
        if ticks_per_second <= 0:
            raise ValueError("CPU clock ticks must be positive")
        self.owner = owner
        self.ticks_per_second = ticks_per_second
        self.previous = None
        self.previous_time = None
        self.owned = set()
        self.busy_samples = 0
        self.reserved_cpus = reserved_cpus
        self.affinity_reader = affinity_reader
        if reserved_cpus is not None and (not reserved_cpus or affinity_reader is None):
            raise ValueError("shared-host accounting requires reserved CPUs and thread affinities")

    def observe(self, current, now):
        # Remember identities so a still-running child remains excluded if it
        # gets reparented. A reused PID is a different process.
        ours = {self.owner} | {pid for pid, start in self.owned
                               if pid in current and current[pid]["start"] == start}
        while True:
            expanded = ours | {pid for pid, row in current.items() if row["ppid"] in ours}
            if expanded == ours:
                break
            ours = expanded
        self.owned = {(pid, current[pid]["start"]) for pid in ours if pid in current}
        if self.previous is None:
            self.previous, self.previous_time = current, now
            return None
        elapsed = now - self.previous_time
        if elapsed <= 0:
            raise ValueError("host CPU samples require increasing monotonic times")
        active = []
        for pid, row in current.items():
            previous = self.previous.get(pid)
            if pid in ours or previous is None or previous["start"] != row["start"]:
                continue
            delta = row["ticks"] - previous["ticks"]
            if delta < 0:
                raise ValueError("host process CPU counter went backwards")
            cores = delta / self.ticks_per_second / elapsed
            if cores:
                active.append({"pid": pid, "ppid": row["ppid"], "name": row["name"], "cores": cores})
        active.sort(key=lambda row: row["cores"], reverse=True)
        # Preserve the tested thresholds, including small services in the raw
        # total. Sustained work on other CPUs can contend for shared memory.
        total = sum(row["cores"] for row in active)
        competing = active if self.reserved_cpus is None else shared_host_processes(
            active, self.reserved_cpus, self.affinity_reader)
        overlap_total = sum(row["cores"] for row in competing)
        busy = bool(competing and (competing[0]["cores"] >= 0.75 or overlap_total >= 1.5))
        self.busy_samples = self.busy_samples + 1 if busy else 0
        self.previous, self.previous_time = current, now
        return {"seconds": elapsed, "unrelated_cpu_cores": total,
                "overlapping_cpu_cores": overlap_total, "processes": active,
                "top_processes": active[:12], "busy": busy, "busy_samples": self.busy_samples,
                "triggered": self.busy_samples >= 2}


class HostGuard:
    """Optional experiment-only monitor; SIGINT lets measured_process clean up.

    The coordination marker remains independent. This guard watches the entire
    host, including other CPUs, and fails closed if monitoring becomes unreliable.
    """

    policy = {"sample_seconds": 2, "sustained_samples": 2,
              "single_process_cpu_cores": 0.75, "total_unrelated_cpu_cores": 1.5}

    def __init__(self, output, *, enabled=True, shared_host=False, reserved_cpus=None):
        self.output = Path(output)
        self.enabled = enabled or shared_host
        self.shared_host = shared_host
        self.scope = "shared_host_disjoint_cpu_affinity" if shared_host else "full_host_quiet" if enabled else "unmonitored"
        self.worker_cpus = sorted(set(reserved_cpus or []))
        if shared_host and not set(self.worker_cpus) <= os.sched_getaffinity(0):
            raise ValueError("shared-host worker CPUs are outside the controller affinity")
        self.reserved_cpus = expand_smt_cpus(self.worker_cpus) if shared_host else None
        self.stopped = threading.Event()
        self.thread = None
        self.reason = None
        self.owner = os.getpid()
        self.state = HostLoadState(self.owner, os.sysconf("SC_CLK_TCK"),
                                  reserved_cpus=self.reserved_cpus,
                                  affinity_reader=process_thread_affinities if shared_host else None)

    def record(self, status):
        write_json(self.output / "host_guard.json", {"enabled": self.enabled, "policy": self.policy,
                   "measurement_scope": self.scope, "shared_host": self.shared_host,
                   "worker_cpus": self.worker_cpus, "reserved_physical_cpus": sorted(self.reserved_cpus or []),
                   "status": status, "reason": self.reason, "timings_accepted": False if self.reason else None})

    def sample(self):
        sample = self.state.observe(process_snapshot(), time.monotonic())
        if sample is not None:
            sample["utc"] = datetime.now(timezone.utc).isoformat()
            sample["measurement_scope"] = self.scope
            if sample["busy"] and not self.shared_host:
                for row in sample["top_processes"]:
                    try:
                        row["cpus"] = sorted(os.sched_getaffinity(row["pid"]))
                    except ProcessLookupError:
                        pass
            with (self.output / "host_samples.jsonl").open("a") as stream:
                stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
        return sample

    def watch(self):
        while not self.stopped.wait(self.policy["sample_seconds"]):
            try:
                sample = self.sample()
                if not sample["triggered"]:
                    continue
                location = " on reserved physical cores" if self.shared_host else ""
                self.reason = f"sustained unrelated CPU work{location}; timings unsuitable for acceptance"
            except Exception as error:
                self.reason = f"host monitoring failed: {error}; timings unsuitable for acceptance"
            try:
                self.record("interrupted")
            finally:
                os.kill(self.owner, signal.SIGINT)
            return

    def start(self):
        if not self.enabled:
            self.record("disabled")
            return
        try:
            self.sample()
            quiet_samples = 0
            while quiet_samples < 2:
                time.sleep(self.policy["sample_seconds"])
                sample = self.sample()
                if sample["triggered"]:
                    raise RuntimeError("host is not quiet; see host_samples.jsonl")
                quiet_samples = 0 if sample["busy"] else quiet_samples + 1
            self.record("monitoring")
        except Exception as error:
            self.reason = f"host preflight failed: {error}; no timing accepted"
            self.record("blocked")
            raise
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join()
        if self.enabled and self.reason is None:
            self.record("finished")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stop()
        if error is not None:
            write_json(self.output / "interruption.json", {"status": "incomplete_unsuitable_for_acceptance",
                       "error": repr(error), "host_guard_reason": self.reason})
        elif self.reason:
            raise RuntimeError(self.reason)


# Existing inner-assignment experiment spelling.
ContentionGuard = HostGuard


def validate_config(config, compile_only=False):
    allowed = {"original", "refactor", "points", "queries", "ground_truth", "metric", "leaf_size",
               "branching", "depth", "niters", "s", "seed", "assignment", "build_threads", "build_cpus",
               "cpus", "threads", "modes", "k", "nprobes", "rerank", "root_m2m", "compression",
               "threshold", "tq8_rerank", "query_limit", "query_seed", "warmup", "repeats",
               "build_blocks", "search_blocks", "recall_tolerance", "python", "compile_plan",
               "original_source", "refactor_source", "search_scopes"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("comparison config contains unknown fields or is not an object")
    if compile_only:
        if not config.get("compile_plan"):
            raise ValueError("--compile-only requires a nonempty compile_plan")
        return
    for key in ("original", "refactor", "points", "queries", "ground_truth", "cpus", "build_cpus"):
        if key not in config:
            raise ValueError(f"comparison requires {key}")
    for key in ("cpus", "build_cpus"):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"{key} must contain unique nonnegative CPU IDs")
    for key in ("threads", "nprobes", "rerank"):
        values = config.get(key, [1])
        if not isinstance(values, list) or not values or any(type(v) is not int or v < (0 if key == "rerank" else 1) for v in values):
            raise ValueError(f"invalid {key} grid")
    for key in ("k", "repeats", "build_blocks", "search_blocks", "build_threads", "leaf_size"):
        if type(config.get(key, 1)) is not int or config.get(key, 1) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if len(config["cpus"]) < max(config.get("threads", [1])) or len(config["build_cpus"]) < config.get("build_threads", 1):
        raise ValueError("provide at least one distinct pinned CPU per thread")
    if any(v and v < config.get("k", 10) for v in config.get("rerank", [20, 50])):
        raise ValueError("rerank must be zero or at least k")
    if config.get("metric", "ip") != "ip":
        raise ValueError("MVIVF retrieval supports only metric ip; L2 remains a clustering objective")
    if config.get("assignment", "float") not in ("float", "tq8"):
        raise ValueError("unsupported assignment mode")
    for key, choices, default in (("modes", ("latency", "batch"), ["latency", "batch"]),
                                   ("compression", ("None", "Wards", "Carve"), ["None", "Wards"])):
        values = config.get(key, default)
        if not isinstance(values, list) or not values or any(v not in choices for v in values):
            raise ValueError(f"unsupported {key}")
    values = config.get("root_m2m", [True])
    if not isinstance(values, list) or not values or any(type(v) is not bool for v in values):
        raise ValueError("root_m2m must be a nonempty list of booleans")
    if type(config.get("tq8_rerank", True)) is not bool:
        raise ValueError("tq8_rerank must be boolean")
    if "python" in config and set(config["python"]) != {"original", "refactor"}:
        raise ValueError("python comparison needs both original and refactor modules/interpreters")
    scopes = config.get("search_scopes", list(SEARCH_SCOPES))
    if not isinstance(scopes, list) or not scopes or len(set(scopes)) != len(scopes) or any(scope not in SEARCH_SCOPES for scope in scopes):
        raise ValueError("search_scopes must contain distinct shared_original/independent_build choices")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": after.st_size}


def check_marker(marker):
    state = json.loads(Path(marker).read_text())
    if state.get("active") is not False:
        raise RuntimeError(f"coordinated experiments are active: {marker}: {state.get('phase', '')}")
    return state


def source_identity(root):
    root = Path(root).resolve(strict=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    names = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "--", "mvsic",
                                     "CMakeLists.txt", "cmake", ".bazelrc", "pyproject.toml", "MODULE.bazel",
                                     "third_party", "tests/refactor"],
                                    cwd=root, text=True).splitlines()
    files = {}
    for name in sorted(set(names)):
        path = root / name
        files[name] = identity(path)["sha256"] if path.is_file() else None
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"path": str(root), "revision": revision, "source_sha256": source_hash, "files": files}


def measured_process(command, directory, *, marker, threads, cpus=None, cwd=None, extra_env=None):
    check_marker(marker)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("commands must be nonempty JSON arrays of strings; shell command strings are unsupported")
    env = os.environ.copy()
    for name in ("PARLAY_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[name] = str(threads)
    env.update(extra_env or {})
    timing = directory / "resource.json"
    timed = ["/usr/bin/time", "-f", '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M}',
             "-o", str(timing)]
    if cpus is not None:
        if not cpus or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("requested CPU affinity is empty or outside the available CPU set")
        timed += ["taskset", "--cpu-list", ",".join(map(str, cpus))]
    timed += command
    start = time.perf_counter()
    cancelled = False
    interruption = None

    def stop(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    with (directory / "stdout.log").open("w") as log:
        process = subprocess.Popen(timed, stdout=log, stderr=subprocess.STDOUT, cwd=cwd,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(0.5)
                try:
                    check_marker(marker)
                except (ValueError, OSError, RuntimeError):
                    cancelled = True
                    stop(process)
                    break
        except BaseException as error:
            interruption = error
            stop(process)
    record = {"command": command, "cwd": str(cwd or Path.cwd()), "threads": threads, "cpus": cpus,
              "wall_seconds": time.perf_counter() - start, "exit_code": process.returncode,
              "coordination_interrupted": cancelled, "controller_interrupted": interruption is not None,
              "extra_environment": extra_env or {}}
    if timing.exists():
        # GNU time may prefix a diagnostic when a process fails or is signalled.
        for line in timing.read_text().splitlines():
            if line.startswith("{"):
                record["resource"] = json.loads(line)
    write_json(directory / "process.json", record)
    if interruption is not None:
        raise interruption
    if cancelled:
        raise RuntimeError("coordination marker became active; process stopped and measurements are invalid")
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); see {directory / 'stdout.log'}")
    return record


def pin_binary(source, directory):
    source_identity = identity(source)
    destination = Path(directory) / source_identity["sha256"] / Path(source).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
        destination.chmod(0o555)
    if identity(destination)["sha256"] != source_identity["sha256"]:
        raise RuntimeError("pinned executable checksum mismatch")
    return destination, {"source": source_identity, "pinned": identity(destination)}


def pin_libraries(source, directory):
    """Pin every resolved MVSIC DSO and override mutable build RPATHs at launch."""
    result = subprocess.run(["ldd", str(source)], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"cannot resolve shared-library dependencies: {result.stderr or result.stdout}")
    libraries = {}
    for line in result.stdout.splitlines():
        if "libmvsic" not in line:
            continue
        match = re.match(r"\s*(libmvsic\S+)\s+=>\s+(\S+)\s+\(", line)
        if not match:
            raise RuntimeError(f"unresolved MVSIC DSO: {line.strip()}")
        libraries[match[1]] = identity(match[2])
    if not libraries:
        return None, {}
    key = hashlib.sha256(json.dumps(libraries, sort_keys=True).encode()).hexdigest()
    pinned_dir = Path(directory) / key
    pinned_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for soname, item in libraries.items():
        target = pinned_dir / soname
        if not target.exists():
            shutil.copy2(item["path"], target)
            target.chmod(0o555)
        pinned = identity(target)
        if pinned["sha256"] != item["sha256"]:
            raise RuntimeError("pinned shared-library checksum mismatch")
        metadata[soname] = {"source": item, "pinned": pinned}
    return pinned_dir, metadata


def library_environment(directory):
    if directory is None:
        return {}
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": str(directory) + (":" + previous if previous else "")}


def verify_pinned_libraries(binary, directory, metadata):
    if directory is None:
        return
    env = {**os.environ, **library_environment(directory)}
    result = subprocess.run(["ldd", str(binary)], env=env, text=True, capture_output=True, check=True)
    found = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(libmvsic\S+)\s+=>\s+(\S+)\s+\(", line)
        if match:
            found[match[1]] = str(Path(match[2]).resolve())
    expected = {soname: item["pinned"]["path"] for soname, item in metadata.items()}
    if found != expected:
        raise RuntimeError(f"loader does not use pinned shared libraries: expected {expected}, found {found}")


def sample_queries(source, output, limit, seed):
    import numpy as np
    path = Path(source)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=3)
    if header.size != 3:
        raise ValueError("truncated PCS header")
    d, n, vectors = map(int, header)
    if not d or not n:
        raise ValueError("empty PCS query data")
    coordinate_bytes = vectors * d * 4
    values = np.memmap(path, mode="r", dtype="<f4", offset=24, shape=(vectors * d,))
    with path.open("rb") as stream:
        stream.seek(24 + coordinate_bytes)
        count = np.fromfile(stream, dtype="<u8", count=1)
        offsets = np.fromfile(stream, dtype="<u8")
    if count.size != 1 or int(count[0]) != n + 1 or offsets.size != n + 1:
        raise ValueError("invalid query PCS offsets")
    if offsets[0] != 0 or int(offsets[-1]) != vectors * d or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("invalid query PCS extents")
    if np.any((offsets[1:] - offsets[:-1]) % d):
        raise ValueError("query PCS cloud sizes do not match dimensions")
    selection = np.arange(n, dtype=np.uint32)
    if limit and limit < n:
        selection = np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False)).astype(np.uint32)
    lengths = offsets[selection + 1] - offsets[selection]
    new_offsets = np.concatenate((np.array([0], dtype=np.uint64), np.cumsum(lengths, dtype=np.uint64)))
    with Path(output).open("wb") as stream:
        np.array([d, len(selection), int(new_offsets[-1]) // d], dtype="<u8").tofile(stream)
        for row in selection:
            values[int(offsets[row]):int(offsets[row + 1])].tofile(stream)
        np.array([len(new_offsets)], dtype="<u8").tofile(stream)
        new_offsets.astype("<u8").tofile(stream)
    np.save(str(output) + ".selection.npy", selection, allow_pickle=False)
    return selection, n


def read_predictions(path):
    import numpy as np
    with Path(path).open("rb") as stream:
        shape = np.fromfile(stream, dtype="<u4", count=2)
        data = np.fromfile(stream, dtype=[("id", "<u4"), ("distance", "<f4")])
    if len(shape) != 2 or data.size != int(shape[0]) * int(shape[1]):
        raise ValueError("invalid prediction file")
    data = data.reshape(tuple(map(int, shape)))
    return data["id"], data["distance"]


def truth_subset(path, selection, nqueries):
    import numpy as np
    with Path(path).open("rb") as stream:
        width = np.fromfile(stream, dtype="<i4", count=1)
    if width.size != 1 or int(width[0]) <= 0 or Path(path).stat().st_size != 4 + nqueries * int(width[0]) * 8:
        raise ValueError("ground truth dimensions do not match query file")
    truth = np.memmap(path, mode="r", dtype=[("distance", "<f4"), ("id", "<u4")],
                      offset=4, shape=(nqueries, int(width[0])))
    return truth[selection]


def recall(ids, truth, k):
    import numpy as np
    if truth.shape[1] < k or len(ids) != len(truth):
        raise ValueError("ground truth does not cover requested queries/k")
    correct = 0
    for row, gold in zip(ids, truth):
        accepted = set(map(int, gold["id"][:k]))
        boundary = float(gold["distance"][k - 1])
        for i in range(k, len(gold)):
            if abs(float(gold["distance"][i]) - boundary) >= 1e-6:
                break
            accepted.add(int(gold["id"][i]))
        returned = set(map(int, row[:k])) - {int(np.iinfo(np.uint32).max)}
        correct += min(k, len(returned & accepted))
    return correct / (len(ids) * k)


def compile_plan(plan, output, marker):
    """Commands run in order; caller supplies isolated clean/incremental build dirs."""
    records = []
    for step in plan:
        if step.get("phase") not in ("configure", "clean", "incremental_public", "incremental_implementation", "link", "wheel"):
            raise ValueError("compile phase must state configure/clean/incremental_public/incremental_implementation/link/wheel")
        # Touches are allowed only in the new clone or an explicitly prepared /tmp
        # snapshot. Never touch baseline headers in the active original checkout.
        touch = step.get("touch")
        if touch:
            path = Path(touch).resolve(strict=True)
            new_root = Path(__file__).resolve().parents[2]
            if not (path.is_relative_to(new_root) or path.is_relative_to(Path("/tmp"))):
                raise ValueError("incremental compilation must use a writable new-tree or /tmp snapshot")
            path.touch()
        label = f'{len(records):02d}-{step["name"]}-{step["phase"]}'
        process = measured_process(step["command"], output / label, marker=marker,
                                   threads=step.get("threads", 1), cpus=step.get("cpus"), cwd=step.get("cwd"))
        record = {**step, "process": process,
                  "outputs": [identity(path) for path in step.get("outputs", [])]}
        records.append(record)
        write_json(output / "compilation.json", records)
    return records


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = json.dumps({"scope": row.get("scope", "shared_original"), "implementation": row["implementation"],
                          "params": row["params"]}, sort_keys=True)
        grouped.setdefault(key, []).append(row)
    result = []
    for group in grouped.values():
        first = group[0]
        result.append({"scope": first.get("scope", "shared_original"),
                       "implementation": first["implementation"], "params": first["params"],
                       "recall_min": min(row["recall"] for row in group),
                       "recall_max": max(row["recall"] for row in group),
                       "qps_median": statistics.median(row["native"]["qps"] for row in group),
                       "trials": len(group),
                       "peak_rss_bytes_median": statistics.median(row["native"]["peak_rss_bytes"] for row in group)})
    return result


def match_recall(rows, tolerance):
    result = []
    for old in (row for row in rows if row["implementation"] == "original"):
        policy = {key: value for key, value in old["params"].items() if key not in ("nprobes", "rerank")}
        options = [row for row in rows if row["implementation"] == "refactor"
                   and row.get("scope", "shared_original") == old.get("scope", "shared_original")
                   and {key: value for key, value in row["params"].items() if key not in ("nprobes", "rerank")} == policy
                   and row["recall_min"] >= old["recall_max"] - tolerance]
        candidate = max(options, key=lambda row: row["qps_median"]) if options else None
        result.append({"original": old, "refactor": candidate, "recall_tolerance": tolerance,
                       "speed_ratio": None if candidate is None else candidate["qps_median"] / old["qps_median"]})
    return result


def search_trials(config, builds):
    """Keep shared-tree isolation and independently built-tree quality separate."""
    first_original = next(row for row in builds if row["implementation"] == "original")
    result = []
    for scope in config.get("search_scopes", SEARCH_SCOPES):
        if scope == "shared_original":
            block = [{"scope": scope, "implementation": kind, "build_number": None,
                      "artifact": first_original["artifact"]}
                     for kind in ("original", "refactor", "refactor", "original")]
        else:
            # builds is itself ABBA ordered, including each independent repetition.
            block = [{"scope": scope, "implementation": row["implementation"], "build_number": number,
                      "artifact": row["artifact"]} for number, row in enumerate(builds)]
        result.extend(block * config.get("search_blocks", 1))
    return result


def compare(config, output, marker):
    import itertools
    check_marker(marker)
    binaries, binary_ids, libraries = {}, {}, {}
    for kind in ("original", "refactor"):
        binaries[kind], binary_ids[kind] = pin_binary(config[kind], output / "bin")
        libraries[kind], binary_ids[kind]["libraries"] = pin_libraries(config[kind], output / "libraries")
        verify_pinned_libraries(binaries[kind], libraries[kind], binary_ids[kind]["libraries"])
    python_modules, python_ids, python_libraries = {}, {}, {}
    if "python" in config:
        for kind in ("original", "refactor"):
            item = config["python"][kind]
            python_modules[kind], module_id = pin_binary(item["module"], output / "python")
            python_libraries[kind], library_ids = pin_libraries(item["module"], output / "libraries")
            verify_pinned_libraries(python_modules[kind], python_libraries[kind], library_ids)
            python_ids[kind] = {"module": module_id, "interpreter": identity(item["executable"]), "libraries": library_ids}
    inputs = {key: identity(config[key]) for key in ("points", "queries", "ground_truth")}
    selection, total_queries = sample_queries(config["queries"], output / "queries.pcs",
                                              config.get("query_limit", 1000), config.get("query_seed", 42))
    truth = truth_subset(config["ground_truth"], selection, total_queries)
    provenance = {"schema_version": 1, "retrieval_metric": "ip", "config": config, "inputs": inputs, "binaries": binary_ids,
                  "sampled_queries": identity(output / "queries.pcs"),
                  "selection": identity(str(output / "queries.pcs") + ".selection.npy"),
                  "marker": check_marker(marker), "platform": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else list(os.uname()),
                  "available_cpus": sorted(os.sched_getaffinity(0)), "python": python_ids}
    provenance["sources"] = {
        "original": source_identity(config.get("original_source", "/home/kishen/MVSIC")),
        "refactor": source_identity(config.get("refactor_source", Path(__file__).resolve().parents[2]))}
    provenance["environment"] = {name: os.environ.get(name) for name in
        ("MVSIC_CPU_ISA", "MVIVF_SUBBATCH", "TSC_GHZ", "LD_LIBRARY_PATH", "LD_PRELOAD",
         "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC", "OMP_WAIT_POLICY", "KMP_AFFINITY",
         "GOMP_CPU_AFFINITY", "GLIBC_TUNABLES", "MALLOC_CONF", "MALLOC_ARENA_MAX")}
    import numpy as np
    provenance["numpy_version"] = np.__version__
    write_json(output / "provenance.json", provenance)
    builds = []
    common = {"metric": config.get("metric", "ip"), "leaf-size": config.get("leaf_size", 100),
              "branching": config.get("branching", 0), "depth": config.get("depth", 0),
              "niters": config.get("niters", 5), "s": config.get("s", 0), "seed": config.get("seed", 0),
              "assignment": config.get("assignment", "float")}
    original_skeleton = None
    build_order = ["original", "refactor", "refactor", "original"] * config.get("build_blocks", 1)
    for number, kind in enumerate(build_order):
        directory = output / f"build-{number:02d}-{kind}"
        skeleton = output / f"build-{number:02d}-{kind}.idx"
        result_path = output / f"build-{number:02d}-{kind}.json"
        options = {**common, "points": config["points"], "index": str(skeleton), "output": str(result_path), "mode": "build"}
        command = [str(binaries[kind])] + [str(item) for pair in options.items() for item in ("--" + pair[0], pair[1])]
        process = measured_process(command, directory, marker=marker, threads=config.get("build_threads", 1),
                                   cpus=config.get("build_cpus"), extra_env=library_environment(libraries[kind]))
        native = json.loads(result_path.read_text())
        builds.append({"implementation": kind, "native": native, "process": process, "artifact": identity(skeleton)})
        if kind == "original" and original_skeleton is None:
            original_skeleton = skeleton
        write_json(output / "builds.json", builds)
    provenance["shared_skeleton"] = identity(original_skeleton)
    trials = search_trials(config, builds)
    provenance["search_trials_per_operating_point"] = trials
    write_json(output / "provenance.json", provenance)
    rows = []
    grid = itertools.product(config.get("modes", ["latency", "batch"]), config.get("threads", [1]),
                             config.get("nprobes", [1, 2, 4, 8, 16, 32, 64]), config.get("rerank", [20, 50]),
                             config.get("root_m2m", [True]), config.get("compression", ["None", "Wards"]))
    for mode, threads, probes, rerank, root_m2m, compression in grid:
        cpus = config.get("cpus")
        if cpus is not None:
            if len(cpus) < threads:
                raise ValueError("supply at least one distinct CPU per configured thread")
            cpus = cpus[:threads]
        params = {"mode": mode, "threads": threads, "nprobes": probes, "rerank": rerank,
                  "root-m2m": int(root_m2m), "compression": compression, "k": config.get("k", 10),
                  "tq8-rerank": int(config.get("tq8_rerank", True)), "threshold": config.get("threshold", 0.5)}
        for trial in trials:
            kind = trial["implementation"]
            label = f'search-{len(rows):05d}-{trial["scope"]}-{kind}'
            result_path = output / f"{label}.json"
            options = {**common, **{key: value for key, value in params.items() if key != "threads"},
                       "points": config["points"], "queries": str(output / "queries.pcs"),
                       "index": trial["artifact"]["path"], "output": str(result_path),
                       "repeats": config.get("repeats", 5), "warmup": config.get("warmup", 10)}
            command = [str(binaries[kind])] + [str(item) for pair in options.items() for item in ("--" + pair[0], pair[1])]
            process = measured_process(command, output / label, marker=marker, threads=threads, cpus=cpus,
                                       extra_env=library_environment(libraries[kind]))
            native = json.loads(result_path.read_text())
            ids, distances = read_predictions(str(result_path) + ".predictions")
            score = recall(ids, truth, params["k"])
            rows.append({**trial, "params": params, "native": native, "recall": score,
                         "predictions": identity(str(result_path) + ".predictions"), "process": process})
            write_json(output / "searches.json", rows)
    summaries = aggregate(rows)
    python_rows = []
    if python_modules:
        worker = Path(__file__).with_name("python_bench.py").resolve()
        for row in rows:
            kind, params = row["implementation"], row["params"]
            label = f'python-{len(python_rows):05d}-{row["scope"]}-{kind}'
            result_path = output / f"{label}.json"
            job = {"implementation": kind, "module": str(python_modules[kind]), "config": config,
                   "params": params, "queries": str(output / "queries.pcs"),
                   "index": row["artifact"]["path"], "output": str(result_path)}
            job_path = output / f"{label}.job.json"
            write_json(job_path, job)
            cpus = config.get("cpus")
            if cpus is not None:
                cpus = cpus[:params["threads"]]
            process = measured_process([config["python"][kind]["executable"], str(worker), "--job", str(job_path)],
                                       output / label, marker=marker, threads=params["threads"], cpus=cpus,
                                       extra_env=library_environment(python_libraries[kind]))
            native = json.loads(result_path.read_text())
            import numpy as np
            with np.load(str(result_path) + ".npz", allow_pickle=False) as arrays:
                score = recall(arrays["ids"], truth, params["k"])
            python_rows.append({"implementation": kind, "scope": row["scope"], "build_number": row["build_number"],
                                "artifact": row["artifact"], "params": params, "native": native, "recall": score,
                                "predictions": identity(str(result_path) + ".npz"), "process": process})
            write_json(output / "python-searches.json", python_rows)
    report = {"schema_version": 1, "scope": "IP retrieval: original TQ4 centers + TQ1 leaves + selected rerank versus compiled facade",
              "search_scopes": {scope: {"shared_original": "both implementations search the first original tree to isolate serving",
                                         "independent_build": "each implementation searches every one of its own fresh builds to assess build quality"}[scope]
                                for scope in config.get("search_scopes", SEARCH_SCOPES)},
              "shared_skeleton": identity(original_skeleton), "aggregation": "median across ABBA processes of minimum repeated native time",
              "builds": builds, "operating_points": summaries,
              "matched_recall": match_recall(summaries, config.get("recall_tolerance", 0.001)),
              "python_operating_points": aggregate(python_rows),
              "python_matched_recall": match_recall(aggregate(python_rows), config.get("recall_tolerance", 0.001)),
              "limitations": ["compiler/wheel times require explicit compile_plan measurements",
                              "Python baseline is a one-specialization IP module with original registration/conversion code, not the full legacy extension",
                              "memory includes corpus, encodings, scratch, runtime and allocator retention",
                              "VNNI/AMX not hardware validated on this host"]}
    write_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--compile-only", action="store_true")
    host = parser.add_mutually_exclusive_group()
    host.add_argument("--quiet-host", action="store_true", help="stop if sustained unrelated CPU work appears")
    host.add_argument("--shared-host", action="store_true", help="allow background work confined away from configured physical cores")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config, args.compile_only)
    worker_cpus = set() if args.compile_only else set(config["cpus"]) | set(config["build_cpus"])
    for step in config.get("compile_plan", []):
        if args.shared_host and not step.get("cpus"):
            raise ValueError("shared-host compilation requires explicit cpus for every compile step")
        worker_cpus.update(step.get("cpus") or [])
    output = Path(args.output).expanduser().resolve()
    if not output.is_relative_to(RESULT_ROOT):
        raise ValueError(f"generated comparison artifacts must be under {RESULT_ROOT}")
    check_marker(args.marker)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    with HostGuard(output, enabled=args.quiet_host, shared_host=args.shared_host, reserved_cpus=worker_cpus):
        if config.get("compile_plan"):
            compile_plan(config["compile_plan"], output / "compilation", args.marker)
        if not args.compile_only:
            compare(config, output, args.marker)
    print(output)


if __name__ == "__main__":
    main()
