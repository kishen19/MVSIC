"""Create build-local dependency overlays without editing source checkouts."""

import argparse
import filecmp
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def write_changed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def copy_tree(source, target, excluded=(), rewrite_eigen=False):
    """Copy headers/sources while keeping identical outputs' timestamps stable."""
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if relative.as_posix() in excluded or "Eigen" in relative.parts:
            continue
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if rewrite_eigen and path.suffix in {".h", ".hpp", ".cpp"}:
            write_changed(output, path.read_text().replace("rabitqlib/third/Eigen/", "Eigen/"))
        elif not output.exists() or not filecmp.cmp(path, output, shallow=False):
            shutil.copyfile(path, output)


def patched_text(text, relative, patch, output):
    # Patch only a temporary copy, then publish the final text if it changed.
    # This keeps a repeated CMake configure from rebuilding unchanged headers.
    with tempfile.TemporaryDirectory(dir=output) as temp:
        root = Path(temp)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        result = subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(patch.resolve())],
            cwd=root, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Cannot apply {patch}:\n{result.stdout}\n{result.stderr}")
        return path.read_text()


def prepare(parlay, rabitq, parlayann, output, patches):
    output.mkdir(parents=True, exist_ok=True)
    copy_tree(parlay / "parlay", output / "parlay/parlay", {"scheduler.h"})
    scheduler = (parlay / "parlay/scheduler.h").read_text()
    if "PARLAY_SHARED_SCHEDULER" not in scheduler:
        scheduler = patched_text(
            scheduler, "parlay/scheduler.h", patches / "parlay/shared_scheduler.patch", output
        )
    elif any(f"internal::shared_scheduler_{operation}(" not in scheduler
             for operation in ("wait", "notify_one", "notify_all")):
        raise RuntimeError(
            "Parlay override has an outdated shared-scheduler patch; "
            "use the original pinned Parlay headers rather than a generated overlay"
        )
    write_changed(output / "parlay/parlay/scheduler.h", scheduler)

    # Public interfaces use the existing ParlayANN point range, but its free
    # functions must be inline when compiled in several isolated backend DSOs.
    point_headers = {"algorithms/utils/euclidian_point.h", "algorithms/utils/mips_point.h"}
    copy_tree(parlayann / "algorithms", output / "parlayann/algorithms",
              {name.removeprefix("algorithms/") for name in point_headers})
    for relative in point_headers:
        text = (parlayann / relative).read_text()
        text = re.sub(r"(?m)^float (euclidian_distance\w*|mips_distance\w*)\(",
                      r"inline float \1(", text)
        write_changed(output / "parlayann" / relative, text)

    rotator_path = "rabitqlib/utils/rotator.hpp"
    copy_tree(rabitq / "include", output / "rabitq/include", {rotator_path}, rewrite_eigen=True)
    copy_tree(rabitq / "src", output / "rabitq/src", {"utils/cpu_features.cpp"}, rewrite_eigen=True)
    rotator = (rabitq / "include" / rotator_path).read_text()
    if "explicit_seed" not in rotator:
        rotator = patched_text(
            rotator, f"include/{rotator_path}", patches / "rabitq/seeded_rotator.patch", output
        )
    rotator = rotator.replace("floor_log2(dim)", "floor_log2(padded_dim)")
    for exponent in range(12, 17):
        if f"case {exponent}:" not in rotator:
            previous = (
                f"            case {exponent-1}:\n"
                f"                this->fht_float_ = helper_float_{exponent-1};\n"
                "                break;\n"
            )
            if previous not in rotator:
                raise RuntimeError("RaBitQ rotator does not match its pinned source")
            rotator = rotator.replace(previous, previous + (
                f"            case {exponent}:\n"
                f"                this->fht_float_ = helper_float_{exponent};\n"
                "                break;\n"
            ))
    write_changed(output / "rabitq/include" / rotator_path, rotator)

    # The pinned dispatcher tests CPUID alone. An OS must also save the needed
    # YMM/ZMM state before those instructions can be used safely.
    cpu = (rabitq / "src/utils/cpu_features.cpp").read_text()
    if "mvsic_os_avx" not in cpu:
        assignment = "    detected.fma = (ecx >> 12) & 1;"
        if assignment not in cpu:
            raise RuntimeError("RaBitQ CPU detection does not match its pinned source")
        cpu = cpu.replace(assignment, """    uint64_t xcr0 = 0;
    if ((ecx & (1u << 27)) != 0) {
        uint32_t lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
    }
    const bool mvsic_os_avx = (ecx & (1u << 28)) != 0 && (xcr0 & 0x6) == 0x6;
    const bool mvsic_os_avx512 = mvsic_os_avx && (xcr0 & 0xe6) == 0xe6;
    detected.fma = mvsic_os_avx && ((ecx >> 12) & 1);""")
        for field, source, bit in (
            ("avx2", "l7_ebx", 5), ("avx512f", "l7_ebx", 16),
            ("avx512dq", "l7_ebx", 17), ("avx512bw", "l7_ebx", 30),
            ("avx512vpopcntdq", "l7_ecx", 14),
        ):
            gate = "mvsic_os_avx" if field == "avx2" else "mvsic_os_avx512"
            cpu = cpu.replace(f"detected.{field} = ({source} >> {bit}) & 1;",
                              f"detected.{field} = {gate} && (({source} >> {bit}) & 1);")
    write_changed(output / "rabitq/src/utils/cpu_features.cpp", cpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("parlay", "rabitq", "parlayann", "output", "patches"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    prepare(**vars(parser.parse_args()))
