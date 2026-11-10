"""Encode BEIR HotpotQA train/dev without rewriting the legacy corpus/test PCS.

Downloads only query text and split qrels. All selection uses official split IDs;
positive relevance labels filter queries but are never used as retrieval GT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from embed.encoders import ColBERTEncoder
from embed.formats import (AtomicTextWriter, StreamingPCSWriter, mmap_pcs_offsets,
                           mmap_pcs_values, read_pcs_layout, validate_pcs, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/beir/hotpotqa'))
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    output = args.output.resolve()
    for split in ('train', 'dev'):
        for suffix in (f'queries_{split}.pcs', f'query_ids_{split}.txt'):
            if (output/f'hotpotqa_{suffix}').exists():
                raise FileExistsError(output/f'hotpotqa_{suffix}')
    source = output/'embedding_source'
    source.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revisions = {repo: api.repo_info(repo, repo_type='dataset').sha
                 for repo in ('BeIR/hotpotqa', 'BeIR/hotpotqa-qrels')}
    def download(repo, name):
        return Path(hf_hub_download(repo, name, repo_type='dataset',
                    revision=revisions[repo], local_dir=source/repo.split('/')[-1]))
    query_path = download('BeIR/hotpotqa', 'queries/queries-00000-of-00001.parquet')
    table = pq.read_table(query_path).to_pydict()
    all_rows = list(zip(map(str, table['_id']), table['text']))
    if len({qid for qid,_ in all_rows}) != len(all_rows):
        raise ValueError('duplicate query IDs')
    positives = {}
    source_hashes = {str(query_path.relative_to(output)): hashlib.sha256(query_path.read_bytes()).hexdigest()}
    for split in ('train', 'dev', 'test'):
        path = download('BeIR/hotpotqa-qrels', split+'.tsv')
        with path.open() as f:
            records = list(csv.DictReader(f, delimiter='\t'))
        positives[split] = {r['query-id'] for r in records if float(r['score']) > 0}
        source_hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = {s: [(qid,text) for qid,text in all_rows if qid in ids] for s,ids in positives.items()}
    for s in rows:
        if len(rows[s]) != len(positives[s]): raise ValueError('missing query IDs: '+s)
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if positives[a] & positives[b]: raise ValueError(f'split leakage: {a}/{b}')
    print('Verified disjoint splits:', {s:len(r) for s,r in rows.items()}, flush=True)
    reference_path = output/'hotpotqa_queries.pcs'
    reference = read_pcs_layout(reference_path)
    if reference.count != len(rows['test']) or reference.dim != 128:
        raise ValueError('legacy test PCS dimensions/count differ from official test')
    values, offsets = mmap_pcs_values(reference), mmap_pcs_offsets(reference)
    # The legacy test file was reordered. Locate a sampled source query by a
    # cheap mean-vector descriptor, then verify all 32x128 original values.
    reference_means = np.stack([values[int(offsets[i])//128:int(offsets[i+1])//128].mean(axis=0)
                                for i in range(reference.count)])
    sample = np.unique(np.linspace(0, reference.count-1, 16, dtype=int))
    started = time.time()
    status = {'status':'starting', 'checkpoint':'colbert-ir/colbertv2.0',
              'source_revisions':revisions,'source_sha256':source_hashes,
              'split_counts':{s:len(r) for s,r in rows.items()}, 'split_overlap':0,
              'query_order':'source query parquet order filtered by positive qrels IDs',
              'legacy_test':str(reference_path),'verified_source_test_rows':sample.tolist(),
              'files':{}}
    manifest = output/'hotpotqa_training_manifest.json'
    write_json(manifest, status)
    with ColBERTEncoder('colbert-ir/colbertv2.0',doc_maxlen=300,query_maxlen=32,
                        threads=args.threads,zero_atol=1e-9) as encoder:
        reproduced = encoder.encode_queries([rows['test'][i][1] for i in sample])
        maximum_error = 0.0
        matched_rows = []
        for i,cloud in zip(sample,reproduced):
            old_row = int(np.square(reference_means-cloud.mean(axis=0)).mean(axis=1).argmin())
            old = values[int(offsets[old_row])//128:int(offsets[old_row+1])//128]
            if old.shape != cloud.shape or not np.allclose(old,cloud,atol=2e-4,rtol=2e-4):
                raise ValueError(f'legacy test reproduction failed at row {i}, shapes {old.shape}/{cloud.shape}')
            matched_rows.append({'source_row':int(i),'query_id':rows['test'][i][0],'legacy_row':old_row})
            maximum_error = max(maximum_error,float(np.max(np.abs(old-cloud))))
        print(f'Legacy test reproduction passed on {len(sample)} rows; max error={maximum_error:g}',flush=True)
        status.update(status='encoding',encoder=encoder.metadata(),test_sample_max_error=maximum_error,
                      verified_legacy_matches=matched_rows,
                      test_alignment_note='16 sampled IDs verified by full embedding comparison; legacy test order differs from source; no full test ID mapping claimed')
        write_json(manifest,status)
        for split in ('train','dev'):
            pcs = output/f'hotpotqa_queries_{split}.pcs'
            ids = output/f'hotpotqa_query_ids_{split}.txt'
            writer = StreamingPCSWriter(pcs,len(rows[split])); id_writer=AtomicTextWriter(ids)
            try:
                for start in range(0,len(rows[split]),args.batch_size):
                    batch=rows[split][start:start+args.batch_size]
                    for (qid,_),cloud in zip(batch,encoder.encode_queries([text for _,text in batch])):
                        writer.append(cloud);id_writer.write_id(qid)
                    if start % (args.batch_size*10)==0:
                        status.update(active_split=split,encoded=start+len(batch),elapsed_seconds=time.time()-started)
                        write_json(manifest,status)
                        print(f'{split}: {start+len(batch)}/{len(rows[split])}, elapsed={time.time()-started:.1f}s',flush=True)
                writer.finalize();id_writer.finalize()
            except Exception:
                writer.abort();id_writer.abort();raise
            status['files'][split]={'pcs':str(pcs),'ids':str(ids),'validation':validate_pcs(pcs)}
            write_json(manifest,status)
            print(f'Completed {pcs}',flush=True)
        status.update(status='complete',elapsed_seconds=time.time()-started)
        write_json(manifest,status)


if __name__=='__main__': main()
