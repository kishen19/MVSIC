#!/usr/bin/env python3
"""Run the YAML-defined leaf quantization ablation with the native harness."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'experiments/vamana_ablations/configs'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='fiqa')
    default_binary = ROOT / 'bazel-bin/mvsic/algorithms/vamana/quant_bench'
    if not default_binary.is_file():
        default_binary = ROOT / 'experiments/vamana_ablations/quantization/bin/bench'
    parser.add_argument('--binary', type=Path,
                        default=default_binary)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT / 'experiments/vamana_ablations/quantization')
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--build-threads', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.repeats < 1 or args.build_threads < 1:
        parser.error('repeats and build-threads must be positive')
    if args.cpu not in os.sched_getaffinity(0):
        parser.error(f'CPU {args.cpu} is outside this process affinity')
    build_path = CONFIGS / 'vamana.quant.build.yaml'
    search_path = CONFIGS / 'vamana.quant.search.yaml'
    build_yaml = yaml.safe_load(build_path.read_text())
    search_yaml = yaml.safe_load(search_path.read_text())
    ds = next(d for d in search_yaml['datasets'] if d['name'] == args.dataset)
    build_ds = next(d for d in build_yaml['datasets'] if d['name'] == args.dataset)
    if (ds['path'], ds['index_dir']) != (build_ds['path'], build_ds['index_dir']):
        raise ValueError('build and search dataset paths differ')
    family = next(i for i in search_yaml['indices'] if i['name'] == 'vamana')
    build_family = next(i for i in build_yaml['indices'] if i['name'] == 'vamana')
    if family['metric'] != 'ip' or build_family['metric'] != 'ip':
        raise ValueError('native runner supports IP only')
    if len(family['builds']) != 1:
        raise ValueError('expected one shared graph')
    build = family['builds'][0]
    build_config = next(b for b in build_family['build_configs'] if b['name'] == build['build_name'])
    params = build['build_params']
    for field in ['R', 'L', 'alpha', 'two_pass']:
        if params[field] != build_config['params'][field]:
            raise ValueError(f'build/search mismatch: {field}')
    if len(build['search_configs']) != 1:
        raise ValueError('expected one search block')
    search = build['search_configs'][0]['params']
    if len(search['k']) != 1 or search['tq8_rerank'] != [True]:
        raise ValueError('expected a single k and TQ8 reranking')
    quantizers = {'TQ': 'tq4', 'OneBitTQ': 'tq1'}
    if {v['quantizer'] for v in build['variants']} != set(quantizers):
        raise ValueError('expected TQ and OneBitTQ leaf variants')
    k = search['k'][0]
    out = args.output_root.resolve() / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    points = ROOT / ds['path'] / f'{args.dataset}_points.pcs'
    queries = ROOT / ds['path'] / f'{args.dataset}_queries.pcs'
    gt = ROOT / ds['path'] / f'{args.dataset}_chamfer_neighbors.gt'
    for path in [binary, points, queries, gt]:
        if not path.is_file():
            raise FileNotFoundError(path)
    index_dir = ROOT / ds['index_dir'] / 'vamana' / build['build_name']
    index_path = index_dir / 'index.bin'
    params_path = index_dir / 'index_params.json'
    if index_path.exists():
        if not params_path.exists():
            raise ValueError(f'existing graph requires build provenance: {params_path}')
        saved = json.loads(params_path.read_text())
        for field, value in params.items():
            if saved.get(field) != value:
                raise ValueError(f'existing graph has different {field}')
    with queries.open('rb') as handle:
        _, query_count = struct.unpack('<QQ', handle.read(16))
    ids = (np.sort(np.random.default_rng(42).choice(query_count, 1000, replace=False))
           if query_count > 1000 else np.arange(query_count))
    query_ids = out / 'query_ids.txt'
    query_text = ''.join(f'{i}\n' for i in ids)
    if query_ids.exists() and query_ids.read_text() != query_text:
        raise ValueError('existing query IDs differ')
    query_ids.write_text(query_text)
    prefix = out / 'shared_quantizer'
    settings = {
        'dataset': args.dataset, 'build_config': build, 'build_yaml_sha256': digest(build_path),
        'search_yaml_sha256': digest(search_path), 'binary_sha256': digest(binary),
        'latency_cpu': args.cpu, 'latency_threads': 1, 'repeats': args.repeats,
        'query_count': len(ids), 'query_cap': 1000, 'query_seed': 42,
        'query_ids_sha256': digest(query_ids), 'mmap': True,
        'early_stop': False, 'same_leaf_rotation': True, 'shared_tq8': True,
        'index_path': str(index_path),
        'inputs': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
                    'mtime_ns': p.stat().st_mtime_ns} for p in [points, queries, gt]},
    }
    settings_path = out / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('experiment settings changed; use a new --output-root')
    write_json(settings_path, settings)
    status_path = out / 'status.json'
    history_path = out / 'commands.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    status = {'state': 'running', 'pid': os.getpid(), 'started_at': time.time(),
              'completed_curves': [], 'expected_curves': 6, 'child_pid': None}
    write_json(status_path, status)

    def execute(command, log, threads, label):
        status.update(current=label, child_pid=None)
        write_json(status_path, status)
        history.append({'argv': command, 'log': str(log), 'threads': threads,
                        'started_at': time.time()})
        write_json(history_path, history)
        print(f'{label}: {log}', flush=True)
        with log.open('w') as stream:
            with subprocess.Popen(command, env=dict(env, PARLAY_NUM_THREADS=str(threads)),
                                  stdout=stream, stderr=subprocess.STDOUT) as child:
                status['child_pid'] = child.pid
                write_json(status_path, status)
                while child.poll() is None:
                    if (out / 'STOP').exists():
                        child.terminate()
                        child.wait()
                        raise RuntimeError('STOP file requested termination')
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
        history[-1]['finished_at'] = time.time()
        history[-1]['returncode'] = 0
        write_json(history_path, history)
        status['child_pid'] = None
        write_json(status_path, status)

    common = [str(binary), '--method', 'mv_vamana', '--points', str(points),
              '--queries', str(queries), '--gt', str(gt), '--index', str(index_path),
              '--R', str(params['R']), '--build-L', str(params['L']),
              '--alpha', str(params['alpha']), '--k', str(k), '--mmap',
              '--query-ids', str(query_ids), '--seed', '42', '--sample', '1000',
              '--quantizer-cache-prefix', str(prefix), '--repeats', str(args.repeats)]
    if params['two_pass']:
        common += ['--two-pass']
    try:
        execute(common + ['--build-only'], out / 'graph_load.log' if index_path.exists()
                else out / 'build.log', args.build_threads, 'shared graph')
        if not params_path.exists():
            write_json(params_path, params)
        graph_hash = digest(index_path)
        if args.build_only:
            status.update(state='build_complete', graph_sha256=graph_hash)
            write_json(status_path, status)
            return
        for rerank in search['config']:
            for variant in build['variants']:
                label = f'{variant["name"]}_{rerank["name"]}'
                directory = out / 'results' / variant['name']
                directory.mkdir(parents=True, exist_ok=True)
                csv_path = directory / f'latency_results_{rerank["name"]}.csv'
                marker = csv_path.with_suffix('.complete.json')
                command = ['taskset', '-c', str(args.cpu)] + common + [
                    '--quant', quantizers[variant['quantizer']], '--num-rerank',
                    str(rerank['num_rerank']), '--tq8-rerank', 'true',
                    '--L', ','.join(map(str, search['L'])), '--csv', str(csv_path)]
                expected = {'command': command, 'graph_sha256': graph_hash,
                            'settings_sha256': digest(settings_path)}
                if marker.exists():
                    recorded = json.loads(marker.read_text())
                    if any(recorded.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'completed curve settings changed: {marker}')
                    if recorded['csv_sha256'] != digest(csv_path):
                        raise ValueError(f'completed CSV changed: {csv_path}')
                    if digest(str(csv_path) + '.query_ids.txt') != digest(query_ids):
                        raise ValueError(f'completed curve query order changed: {csv_path}')
                    for name, checksum in recorded['models'].items():
                        if digest(out / name) != checksum:
                            raise ValueError(f'completed curve model changed: {name}')
                else:
                    execute(command, directory / f'{rerank["name"]}.log', 1, label)
                    rows = list(csv.DictReader(csv_path.open()))
                    grid = {(int(r['repeat']), int(r['L'])) for r in rows}
                    if len(rows) != args.repeats * len(search['L']) or grid != {
                            (r, width) for r in range(args.repeats) for width in search['L']}:
                        raise ValueError(f'incomplete curve: {csv_path}')
                    if any(int(r['num_queries']) != len(ids) for r in rows):
                        raise ValueError(f'query count changed: {csv_path}')
                    if digest(str(csv_path) + '.query_ids.txt') != digest(query_ids):
                        raise ValueError(f'query order changed: {csv_path}')
                    expected['csv_sha256'] = digest(csv_path)
                    expected['models'] = {p.name: digest(p) for p in out.glob('shared_quantizer.tq*')}
                    write_json(marker, expected)
                status['completed_curves'].append(label)
                write_json(status_path, status)
        if digest(str(prefix) + '.tq1') != digest(str(prefix) + '.tq4'):
            raise ValueError('leaf bit depths did not share the same rotation')
        status.update(state='complete', finished_at=time.time(), graph_sha256=graph_hash,
                      models={p.name: digest(p) for p in out.glob('shared_quantizer.tq*')})
        write_json(status_path, status)
        print('All six full sweeps complete.', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=str(exc), child_pid=None)
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
