#!/usr/bin/env bash
#
# End-to-end IGP setup for the *current* Python environment.
# - Clones DBGroup-SUSTech/multi-vector-retrieval
# - Applies small CMake portability patches (idempotent)
# - Builds the pybind module (CPU path)
# - Installs IGP*.so into current env's site-packages
#
# Usage:
#   bash setup_igp.sh
#   bash setup_igp.sh --repo-dir "$HOME/multi-vector-retrieval" --keep-artifacts
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=""
BUILD_DIR=""
BRANCH=""
PYTHON_BIN="${PYTHON:-python3}"
KEEP_ARTIFACTS=0
TEMP_REPO_DIR=0
SUCCESS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --keep-artifacts) KEEP_ARTIFACTS=1; shift;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

cleanup() {
  # Clean up only after a successful install unless --keep-artifacts.
  if [[ "${SUCCESS}" -eq 1 && "${KEEP_ARTIFACTS}" -eq 0 && "${TEMP_REPO_DIR}" -eq 1 ]]; then
    rm -rf "${REPO_DIR}"
    echo "[setup_igp] cleaned up temporary source/build: ${REPO_DIR}"
  fi
}
trap cleanup EXIT

if [[ -z "${REPO_DIR}" ]]; then
  REPO_DIR="$(mktemp -d "${TMPDIR:-/tmp}/igp-src.XXXXXX")"
  TEMP_REPO_DIR=1
fi

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${REPO_DIR}/build"
fi

echo "[setup_igp] root:      ${ROOT_DIR}"
echo "[setup_igp] repo:      ${REPO_DIR}"
echo "[setup_igp] build:     ${BUILD_DIR}"
echo "[setup_igp] python:    ${PYTHON_BIN}"
echo "[setup_igp] cleanup:   $([[ "${KEEP_ARTIFACTS}" -eq 1 ]] && echo "disabled (--keep-artifacts)" || echo "enabled (default)")"

if ! command -v git >/dev/null 2>&1; then
  echo "[error] git not found" >&2
  exit 2
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "[error] cmake not found (install cmake first)" >&2
  exit 2
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "[error] ninja not found (install ninja first)" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c "import sys; print(sys.version)" >/dev/null 2>&1; then
  echo "[error] python executable is not usable: ${PYTHON_BIN}" >&2
  exit 2
fi

# pybind11 is needed to configure CMake for the Python module.
if ! "${PYTHON_BIN}" -c "import pybind11" >/dev/null 2>&1; then
  echo "[setup_igp] installing pybind11 into current env..."
  "${PYTHON_BIN}" -m pip install pybind11
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[setup_igp] cloning multi-vector-retrieval..."
  if [[ -n "${BRANCH}" ]]; then
    git clone --branch "${BRANCH}" --single-branch \
      https://github.com/DBGroup-SUSTech/multi-vector-retrieval "${REPO_DIR}"
  else
    git clone https://github.com/DBGroup-SUSTech/multi-vector-retrieval "${REPO_DIR}"
  fi
else
  echo "[setup_igp] reusing existing clone."
  git -C "${REPO_DIR}" fetch --all --tags --prune >/dev/null 2>&1 || true
fi

# Patch CMakeLists.txt in an idempotent way to:
# 1) default to CPU builds
# 2) use flexible pybind11 path discovery
# 3) prefer header-only spdlog to avoid runtime libstdc++ / CXXABI mismatches
echo "[setup_igp] patching CMakeLists.txt (idempotent)..."
"${PYTHON_BIN}" - "${REPO_DIR}/CMakeLists.txt" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
orig = text

# CPU-default toggle: replace hard-coded USE_CUDA ON.
text = text.replace(
    "set(USE_CUDA ON)",
    'option(USE_CUDA "Build CUDA-enabled modules/executables" OFF)'
)

# Flexible pybind11 lookup, if old hard-coded block exists.
old_pybind = (
    'set(pybind11_DIR "${CMAKE_PREFIX_PATH}/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")\n'
    "find_package(pybind11 REQUIRED)\n"
)
if old_pybind in text:
    text = text.replace(
        old_pybind,
        'if (DEFINED ENV{PYBIND11_DIR})\n'
        '    set(pybind11_DIR "$ENV{PYBIND11_DIR}")\n'
        "endif ()\n"
        "find_package(pybind11 REQUIRED)\n",
    )

# Flexible spdlog discovery + header-only preference.
if "find_package(spdlog REQUIRED)" in text and "SPDLOG_TARGET" not in text:
    text = text.replace(
        "find_package(spdlog REQUIRED)",
        "find_package(spdlog REQUIRED)\n"
        "if (TARGET spdlog::spdlog_header_only)\n"
        "    set(SPDLOG_TARGET spdlog::spdlog_header_only)\n"
        "else ()\n"
        "    set(SPDLOG_TARGET spdlog::spdlog)\n"
        "endif ()",
    )
    text = text.replace(
        "spdlog::spdlog",
        "${SPDLOG_TARGET}",
    )

if text != orig:
    path.write_text(text, encoding="utf-8")
    print("patched")
else:
    print("already up-to-date")
PY

mkdir -p "${BUILD_DIR}"

PYBIND11_CMAKE_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)"
SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [[ -z "${SITE_PACKAGES}" ]]; then
  echo "[error] could not resolve site-packages for ${PYTHON_BIN}" >&2
  exit 2
fi

CONDA_CMAKE_PREFIX=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CONDA_CMAKE_PREFIX="${CONDA_PREFIX}"
fi

echo "[setup_igp] configuring cmake..."
cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DUSE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
  ${CONDA_CMAKE_PREFIX:+-DCMAKE_PREFIX_PATH="${CONDA_CMAKE_PREFIX}"}

echo "[setup_igp] building IGP pybind module..."
cmake --build "${BUILD_DIR}" --target IGP -j

MODULE_PATH="$("${PYTHON_BIN}" - "${BUILD_DIR}" <<'PY'
import pathlib
import sys
b = pathlib.Path(sys.argv[1])
cands = sorted([p for p in b.rglob("IGP*.so") if p.is_file()])
print(cands[-1] if cands else "")
PY
)"
if [[ -z "${MODULE_PATH}" || ! -f "${MODULE_PATH}" ]]; then
  echo "[error] could not find built IGP*.so under ${BUILD_DIR}" >&2
  exit 2
fi

echo "[setup_igp] installing module:"
echo "  ${MODULE_PATH}"
echo "  -> ${SITE_PACKAGES}/"
cp -f "${MODULE_PATH}" "${SITE_PACKAGES}/"

echo "[setup_igp] verifying import..."
"${PYTHON_BIN}" - <<'PY'
import IGP
print("IGP import ok:", IGP.__file__)
PY

SUCCESS=1
echo "[setup_igp] done."
