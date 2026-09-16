"""Thin, lazy-loading wrappers around ColBERT and ColPali."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


def clean_embedding(
    value: Any,
    *,
    normalize: bool = True,
    zero_atol: float = 1e-9,
) -> np.ndarray:
    """Convert one model result to a finite, unpadded float32 point cloud."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D embedding, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("model produced non-finite embedding values")
    norms = np.linalg.norm(array, axis=1)
    keep = norms > zero_atol
    array = array[keep]
    norms = norms[keep]
    if array.shape[0] == 0:
        raise ValueError("all vectors were padding/zero vectors")
    if normalize:
        array = array / norms[:, None]
    return np.ascontiguousarray(array, dtype=np.float32)


def _embedding_list(output: Any, expected: int) -> list[Any]:
    if hasattr(output, "embeddings"):
        output = output.embeddings
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        result = output
    elif hasattr(output, "ndim") and output.ndim == 3:
        result = [output[index] for index in range(output.shape[0])]
    elif hasattr(output, "dim") and output.dim() == 3:
        result = [output[index] for index in range(output.size(0))]
    else:
        raise ValueError(f"unexpected embedding container {type(output)!r}")
    if len(result) != expected:
        raise ValueError(f"encoder returned {len(result)} items for a batch of {expected}")
    return result


class Encoder(ABC):
    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        pass

    @abstractmethod
    def encode_documents(self, values: Sequence[Any]) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, values: Sequence[Any]) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class ColBERTEncoder(Encoder):
    def __init__(
        self,
        checkpoint: str,
        *,
        doc_maxlen: int,
        query_maxlen: int,
        threads: int | None,
        zero_atol: float,
    ):
        try:
            import torch
            from colbert.infra import ColBERTConfig, Run, RunConfig
            from colbert.modeling.checkpoint import Checkpoint
        except ImportError as error:
            raise RuntimeError("ColBERT generation requires `pip install colbert-ai`") from error

        self._torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.checkpoint = checkpoint
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.zero_atol = zero_atol
        self._context = Run().context(RunConfig(nranks=1, experiment="mvsic_embeddings"))
        self._context.__enter__()
        config = ColBERTConfig(doc_maxlen=doc_maxlen, query_maxlen=query_maxlen)
        self._model = Checkpoint(checkpoint, colbert_config=config)

    def _encode(self, values: Sequence[str], *, queries: bool) -> list[np.ndarray]:
        with self._torch.no_grad():
            if queries:
                output = self._model.queryFromText(
                    list(values), bsize=len(values), to_cpu=True
                )
            else:
                # keep_dims=False uses ColBERT's document attention mask to
                # return a ragged list, so document padding never enters PCS.
                output = self._model.docFromText(
                    list(values),
                    bsize=len(values),
                    keep_dims=False,
                    to_cpu=True,
                    showprogress=False,
                )
        return [
            clean_embedding(item, zero_atol=self.zero_atol)
            for item in _embedding_list(output, len(values))
        ]

    def encode_documents(self, values: Sequence[Any]) -> list[np.ndarray]:
        return self._encode(values, queries=False)

    def encode_queries(self, values: Sequence[Any]) -> list[np.ndarray]:
        return self._encode(values, queries=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "colbert",
            "checkpoint": self.checkpoint,
            "doc_maxlen": self.doc_maxlen,
            "query_maxlen": self.query_maxlen,
            "padding": f"remove rows with L2 norm <= {self.zero_atol:g}",
            "normalization": "row-wise L2",
        }

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None


class ColPaliEncoder(Encoder):
    def __init__(
        self,
        checkpoint: str,
        *,
        dtype: str,
        threads: int | None,
        zero_atol: float,
    ):
        try:
            import torch
            from transformers import ColPaliForRetrieval, ColPaliProcessor
        except ImportError as error:
            raise RuntimeError(
                "ColPali generation requires a Transformers release with ColPali support"
            ) from error

        self._torch = torch
        if threads:
            torch.set_num_threads(threads)
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.zero_atol = zero_atol
        self._model = ColPaliForRetrieval.from_pretrained(
            checkpoint, dtype=dtypes[dtype], device_map="auto"
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(checkpoint)

    def _model_device(self):
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return self._model.device

    def _encode(self, values: Sequence[Any], *, queries: bool) -> list[np.ndarray]:
        if queries:
            inputs = self._processor(text=list(values), return_tensors="pt")
        else:
            inputs = self._processor(images=list(values), return_tensors="pt")
        inputs = inputs.to(self._model_device())
        with self._torch.no_grad():
            output = self._model(**inputs)
        return [
            clean_embedding(item, zero_atol=self.zero_atol)
            for item in _embedding_list(output, len(values))
        ]

    def encode_documents(self, values: Sequence[Any]) -> list[np.ndarray]:
        return self._encode(values, queries=False)

    def encode_queries(self, values: Sequence[Any]) -> list[np.ndarray]:
        return self._encode(values, queries=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "colpali",
            "checkpoint": self.checkpoint,
            "dtype": self.dtype,
            "padding": f"remove rows with L2 norm <= {self.zero_atol:g}",
            "normalization": "row-wise L2",
        }


def create_encoder(
    suite: str,
    checkpoint: str,
    *,
    doc_maxlen: int,
    query_maxlen: int,
    dtype: str,
    threads: int | None,
    zero_atol: float,
) -> Encoder:
    if suite in ("beir", "lotte"):
        return ColBERTEncoder(
            checkpoint,
            doc_maxlen=doc_maxlen,
            query_maxlen=query_maxlen,
            threads=threads,
            zero_atol=zero_atol,
        )
    if suite == "vidore":
        return ColPaliEncoder(
            checkpoint,
            dtype=dtype,
            threads=threads,
            zero_atol=zero_atol,
        )
    raise ValueError(f"unsupported dataset suite {suite!r}")
