#!/usr/bin/env bash
#
# Setup external baselines for MVSIC (select with flags).
#
#   bash setup_external.sh --gem --hnswlib
#   bash setup_external.sh --all
#   bash setup_external.sh                    # same as --all
#   bash setup_external.sh --igp --keep-artifacts
#   bash setup_external.sh --gem --bazel-opt=-c=dbg
#
# Components:
#   --gem        Bazel-build gem_runner (sigmod26gem via MODULE.bazel)
#   --hnswlib    Bazel-build hnswlib_runner
#   --igp        Clone/build/install IGP pybind module (current Python env)
#   --fastplaid  pip install fast_plaid (from requirements.txt pin)
#   --all        Enable all of the above
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SETUP_GEM=0
SETUP_IGP=0
SETUP_FASTPLAID=0
SETUP_HNSWLIB=0
BAZEL_OPTS=()

# IGP-only options (forwarded when --igp is selected).
IGP_REPO_DIR=""
IGP_BUILD_DIR=""
IGP_BRANCH=""
IGP_PYTHON="${PYTHON:-python3}"
IGP_KEEP_ARTIFACTS=0

usage() {
  sed -n '2,22p' "$0"
  echo ""
  echo "Examples:"
  echo "  bash setup_external.sh --gem --hnswlib"
  echo "  bash setup_external.sh --igp --repo-dir \"\$HOME/multi-vector-retrieval\" --keep-artifacts"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gem) SETUP_GEM=1; shift;;
    --igp) SETUP_IGP=1; shift;;
    --fastplaid) SETUP_FASTPLAID=1; shift;;
    --hnswlib) SETUP_HNSWLIB=1; shift;;
    --all)
      SETUP_GEM=1
      SETUP_IGP=1
      SETUP_FASTPLAID=1
      SETUP_HNSWLIB=1
      shift
      ;;
    --bazel-opt=*)
      BAZEL_OPTS+=("${1#*=}")
      shift
      ;;
    --repo-dir) IGP_REPO_DIR="$2"; shift 2;;
    --build-dir) IGP_BUILD_DIR="$2"; shift 2;;
    --branch) IGP_BRANCH="$2"; shift 2;;
    --python) IGP_PYTHON="$2"; shift 2;;
    --keep-artifacts) IGP_KEEP_ARTIFACTS=1; shift;;
    -h|--help) usage 0;;
    *)
      echo "[setup_external] unknown argument: $1" >&2
      usage 2
      ;;
  esac
done

if [[ "$SETUP_GEM$SETUP_IGP$SETUP_FASTPLAID$SETUP_HNSWLIB" == "0000" ]]; then
  SETUP_GEM=1
  SETUP_IGP=1
  SETUP_FASTPLAID=1
  SETUP_HNSWLIB=1
fi

setup_gem() {
  echo "[setup_external/gem] bazel will fetch @sigmod26gem per MODULE.bazel."
  echo "[setup_external/gem] bazel build //benchmarks/gem:gem_runner ${BAZEL_OPTS[*]:-}"
  bazel build "${BAZEL_OPTS[@]}" //benchmarks/gem:gem_runner

  local bin="$ROOT_DIR/bazel-bin/benchmarks/gem/gem_runner"
  if [[ ! -f "$bin" ]]; then
    echo "[setup_external/gem] error: expected $bin after build" >&2
    exit 1
  fi
  "$bin" --help >/dev/null
  echo "[setup_external/gem] done ($bin)."
}

setup_hnswlib() {
  echo "[setup_external/hnswlib] bazel build //benchmarks/hnswlib:hnswlib_runner ${BAZEL_OPTS[*]:-}"
  bazel build "${BAZEL_OPTS[@]}" //benchmarks/hnswlib:hnswlib_runner

  local bin="$ROOT_DIR/bazel-bin/benchmarks/hnswlib/hnswlib_runner"
  if [[ ! -f "$bin" ]]; then
    echo "[setup_external/hnswlib] error: expected $bin after build" >&2
    exit 1
  fi
  "$bin" --help >/dev/null
  echo "[setup_external/hnswlib] done ($bin)."
}

setup_fastplaid() {
  local pin="fast_plaid~=1.4.4.290"
  echo "[setup_external/fastplaid] pip install ${pin}"
  "${IGP_PYTHON}" -m pip install "${pin}"
  echo "[setup_external/fastplaid] verifying import..."
  "${IGP_PYTHON}" -c "from fast_plaid.search import FastPlaid; print('fast_plaid ok:', FastPlaid)"
  echo "[setup_external/fastplaid] done."
}

setup_igp() {
  local repo_dir="$IGP_REPO_DIR"
  local build_dir="$IGP_BUILD_DIR"
  local branch="$IGP_BRANCH"
  local python_bin="$IGP_PYTHON"
  local keep_artifacts="$IGP_KEEP_ARTIFACTS"
  local temp_repo_dir=0
  local success=0

  cleanup_igp() {
    if [[ "${success}" -eq 1 && "${keep_artifacts}" -eq 0 && "${temp_repo_dir}" -eq 1 ]]; then
      rm -rf "${repo_dir}"
      echo "[setup_external/igp] cleaned up temporary source/build: ${repo_dir}"
    fi
  }
  trap cleanup_igp EXIT

  if [[ -z "${repo_dir}" ]]; then
    repo_dir="$(mktemp -d "${TMPDIR:-/tmp}/igp-src.XXXXXX")"
    temp_repo_dir=1
  fi
  if [[ -z "${build_dir}" ]]; then
    build_dir="${repo_dir}/build"
  fi

  echo "[setup_external/igp] root:      ${ROOT_DIR}"
  echo "[setup_external/igp] repo:      ${repo_dir}"
  echo "[setup_external/igp] build:     ${build_dir}"
  echo "[setup_external/igp] python:    ${python_bin}"
  echo "[setup_external/igp] cleanup:   $([[ "${keep_artifacts}" -eq 1 ]] && echo "disabled (--keep-artifacts)" || echo "enabled (default)")"

  for cmd in git cmake ninja; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "[setup_external/igp] error: ${cmd} not found" >&2
      exit 2
    fi
  done
  if ! "${python_bin}" -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo "[setup_external/igp] error: python not usable: ${python_bin}" >&2
    exit 2
  fi
  if ! "${python_bin}" -c "import pybind11" >/dev/null 2>&1; then
    echo "[setup_external/igp] installing pybind11..."
    "${python_bin}" -m pip install pybind11
  fi

  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "[setup_external/igp] cloning multi-vector-retrieval..."
    if [[ -n "${branch}" ]]; then
      git clone --branch "${branch}" --single-branch \
        https://github.com/DBGroup-SUSTech/multi-vector-retrieval "${repo_dir}"
    else
      git clone https://github.com/DBGroup-SUSTech/multi-vector-retrieval "${repo_dir}"
    fi
  else
    echo "[setup_external/igp] reusing existing clone."
    git -C "${repo_dir}" fetch --all --tags --prune >/dev/null 2>&1 || true
  fi

  echo "[setup_external/igp] patching CMakeLists.txt (idempotent)..."
  "${python_bin}" - "${repo_dir}/CMakeLists.txt" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
orig = text

text = text.replace(
    "set(USE_CUDA ON)",
    'option(USE_CUDA "Build CUDA-enabled modules/executables" OFF)',
)

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
    text = text.replace("spdlog::spdlog", "${SPDLOG_TARGET}")

if text != orig:
    path.write_text(text, encoding="utf-8")
    print("patched")
else:
    print("already up-to-date")
PY

  mkdir -p "${build_dir}"
  local pybind11_cmake_dir site_packages
  pybind11_cmake_dir="$("${python_bin}" -m pybind11 --cmakedir)"
  site_packages="$("${python_bin}" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  if [[ -z "${site_packages}" ]]; then
    echo "[setup_external/igp] error: could not resolve site-packages" >&2
    exit 2
  fi

  local conda_prefix="${CONDA_PREFIX:-}"
  echo "[setup_external/igp] configuring cmake..."
  cmake -S "${repo_dir}" -B "${build_dir}" -G Ninja \
    -DUSE_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="${pybind11_cmake_dir}" \
    ${conda_prefix:+-DCMAKE_PREFIX_PATH="${conda_prefix}"}

  echo "[setup_external/igp] building IGP pybind module..."
  cmake --build "${build_dir}" --target IGP -j

  local module_path
  module_path="$("${python_bin}" - "${build_dir}" <<'PY'
import pathlib
import sys
b = pathlib.Path(sys.argv[1])
cands = sorted([p for p in b.rglob("IGP*.so") if p.is_file()])
print(cands[-1] if cands else "")
PY
)"
  if [[ -z "${module_path}" || ! -f "${module_path}" ]]; then
    echo "[setup_external/igp] error: could not find built IGP*.so under ${build_dir}" >&2
    exit 2
  fi

  echo "[setup_external/igp] installing ${module_path} -> ${site_packages}/"
  cp -f "${module_path}" "${site_packages}/"
  "${python_bin}" -c "import IGP; print('IGP import ok:', IGP.__file__)"
  success=1
  echo "[setup_external/igp] done."
}

echo "[setup_external] root: ${ROOT_DIR}"
selected=()
[[ "$SETUP_GEM" -eq 1 ]] && selected+=("gem")
[[ "$SETUP_IGP" -eq 1 ]] && selected+=("igp")
[[ "$SETUP_FASTPLAID" -eq 1 ]] && selected+=("fastplaid")
[[ "$SETUP_HNSWLIB" -eq 1 ]] && selected+=("hnswlib")
echo "[setup_external] selected: ${selected[*]}"

[[ "$SETUP_GEM" -eq 1 ]] && setup_gem
[[ "$SETUP_HNSWLIB" -eq 1 ]] && setup_hnswlib
[[ "$SETUP_FASTPLAID" -eq 1 ]] && setup_fastplaid
[[ "$SETUP_IGP" -eq 1 ]] && setup_igp

echo "[setup_external] all requested components finished."
