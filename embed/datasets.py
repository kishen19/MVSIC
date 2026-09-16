"""Dataset adapters used by the embedding pipeline.

Adapters expose a single ordered corpus and one or more ordered query sets.
External IDs are retained until the pipeline maps them to MVSIC row IDs.
Heavy third-party packages are imported only by the adapter that needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class Record:
    id: str
    value: Any


@dataclass(frozen=True)
class RecordCollection:
    count: int
    iterator_factory: Callable[[], Iterator[Record]]

    def __iter__(self) -> Iterator[Record]:
        return self.iterator_factory()

    def limited(self, limit: int | None) -> "RecordCollection":
        if limit is None or limit >= self.count:
            return self
        if limit <= 0:
            raise ValueError("--limit must be positive")

        def iterator() -> Iterator[Record]:
            for index, record in enumerate(self):
                if index >= limit:
                    break
                yield record

        return RecordCollection(count=limit, iterator_factory=iterator)


@dataclass
class QuerySet:
    name: str
    records: list[Record]
    # None means the source has no judgments. An empty dictionary means the
    # source has judgments but no query currently maps to a positive result.
    qrels: dict[str, list[str]] | None


@dataclass
class DatasetBundle:
    suite: str
    name: str
    source: str
    split: str
    corpus: RecordCollection
    query_sets: list[QuerySet]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_beir(
    name: str,
    output_dir: Path,
    *,
    split: str = "test",
    limit: int | None = None,
) -> DatasetBundle:
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as error:
        raise RuntimeError("BEIR generation requires `pip install beir`") from error

    corpus_file = output_dir / "corpus.jsonl"
    dataset_dir = output_dir
    if not corpus_file.exists():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        dataset_dir = Path(util.download_and_unzip(url, str(output_dir.parent)))
        if not dataset_dir.exists():
            raise ValueError(f"BEIR download did not create the expected dataset directory: {dataset_dir}")

    corpus, queries, qrels_raw = GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)
    corpus_items = list(corpus.items())
    query_items = list(queries.items())
    if limit is not None:
        corpus_items = corpus_items[:limit]
        query_items = query_items[:limit]

    def corpus_iterator() -> Iterator[Record]:
        for document_id, document in corpus_items:
            title = document.get("title", "") or ""
            text = document.get("text", "") or ""
            yield Record(str(document_id), f"{title} {text}".strip())

    qrels: dict[str, list[str]] = {}
    for query_id, documents in qrels_raw.items():
        positive = [str(doc_id) for doc_id, score in documents.items() if float(score) > 0]
        qrels[str(query_id)] = positive

    return DatasetBundle(
        suite="beir",
        name=name,
        source=f"beir/{name}",
        split=split,
        corpus=RecordCollection(len(corpus_items), corpus_iterator),
        query_sets=[
            QuerySet(
                name="search",
                records=[Record(str(query_id), text) for query_id, text in query_items],
                qrels=qrels,
            )
        ],
        metadata={"loader": "beir.GenericDataLoader"},
    )


def load_lotte(
    name: str,
    *,
    split: str = "dev",
    limit: int | None = None,
) -> DatasetBundle:
    try:
        import ir_datasets
    except ImportError as error:
        raise RuntimeError(
            "LoTTE generation requires `pip install ir-datasets`; this adapter uses "
            "the canonical LoTTE hierarchy, including the pooled dataset"
        ) from error

    corpus_source = f"lotte/{name}/{split}"
    try:
        corpus_dataset = ir_datasets.load(corpus_source)
    except Exception as error:
        raise ValueError(f"could not load LoTTE dataset {corpus_source!r}") from error

    corpus_count = corpus_dataset.docs_count()
    if corpus_count is None:
        raise ValueError(f"{corpus_source}: document count is unavailable")

    def corpus_iterator() -> Iterator[Record]:
        for document in corpus_dataset.docs_iter():
            yield Record(str(document.doc_id), document.text)

    corpus = RecordCollection(int(corpus_count), corpus_iterator).limited(limit)
    query_sets: list[QuerySet] = []
    for kind in ("search", "forum"):
        source = f"{corpus_source}/{kind}"
        dataset = ir_datasets.load(source)
        records = [Record(str(query.query_id), query.text) for query in dataset.queries_iter()]
        if limit is not None:
            records = records[:limit]
        qrels: dict[str, list[str]] = {}
        for qrel in dataset.qrels_iter():
            if int(qrel.relevance) > 0:
                qrels.setdefault(str(qrel.query_id), []).append(str(qrel.doc_id))
        query_sets.append(QuerySet(name=kind, records=records, qrels=qrels))

    return DatasetBundle(
        suite="lotte",
        name=name,
        source=corpus_source,
        split=split,
        corpus=corpus,
        query_sets=query_sets,
        metadata={"loader": "ir_datasets", "query_kinds": ["search", "forum"]},
    )


def _field(row: dict[str, Any], candidates: tuple[str, ...], description: str) -> Any:
    for name in candidates:
        if name in row:
            return row[name]
    raise KeyError(f"could not find {description}; available fields: {sorted(row)}")


def load_vidore(
    name: str,
    *,
    source: str | None = None,
    split: str = "test",
    limit: int | None = None,
) -> DatasetBundle:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as error:
        raise RuntimeError("ViDoRe generation requires `pip install datasets`") from error

    source = source or f"vidore/{name}"
    corpus_dataset = load_dataset(source, "corpus", split=split)
    query_dataset = load_dataset(source, "queries", split=split)

    corpus_count = min(len(corpus_dataset), limit) if limit is not None else len(corpus_dataset)

    def corpus_iterator() -> Iterator[Record]:
        for index, row in enumerate(corpus_dataset):
            if index >= corpus_count:
                break
            document_id = _field(
                row, ("corpus-id", "corpus_id", "doc-id", "doc_id", "_id", "id"), "corpus ID"
            )
            image = _field(row, ("image", "page_image"), "document image")
            yield Record(str(document_id), image)

    query_rows = list(query_dataset.select(range(min(len(query_dataset), limit)))) if limit else list(query_dataset)
    records = [
        Record(
            str(_field(row, ("query-id", "query_id", "qid", "_id", "id"), "query ID")),
            _field(row, ("query", "text", "question", "query_text"), "query text"),
        )
        for row in query_rows
    ]

    qrels = None
    configs = get_dataset_config_names(source)
    if "qrels" in configs:
        qrel_dataset = load_dataset(source, "qrels", split=split)
        qrels = {}
        for row in qrel_dataset:
            query_id = _field(row, ("query-id", "query_id", "qid"), "qrel query ID")
            document_id = _field(
                row, ("corpus-id", "corpus_id", "doc-id", "doc_id"), "qrel corpus ID"
            )
            score = _field(row, ("score", "relevance"), "qrel score")
            if float(score) > 0:
                qrels.setdefault(str(query_id), []).append(str(document_id))

    return DatasetBundle(
        suite="vidore",
        name=name,
        source=source,
        split=split,
        corpus=RecordCollection(corpus_count, corpus_iterator),
        query_sets=[QuerySet(name="search", records=records, qrels=qrels)],
        metadata={"loader": "huggingface.datasets"},
    )


def load_bundle(
    suite: str,
    name: str,
    output_dir: Path,
    *,
    source: str | None,
    split: str,
    limit: int | None,
) -> DatasetBundle:
    if suite == "beir":
        return load_beir(name, output_dir, split=split, limit=limit)
    if suite == "lotte":
        return load_lotte(name, split=split, limit=limit)
    if suite == "vidore":
        return load_vidore(name, source=source, split=split, limit=limit)
    raise ValueError(f"unsupported dataset suite {suite!r}")
