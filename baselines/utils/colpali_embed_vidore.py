# colpali_embed_vidore.py
import argparse, os, struct, logging, tempfile, shutil
from typing import List, Tuple
import numpy as np
import torch
from datasets import load_dataset
from transformers import ColPaliForRetrieval, ColPaliProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------- Streaming binary writer ---------------------------


class StreamingBinaryWriter:
    """
    Streams multi-vector embeddings to C++ format:
      [Q] dim, [Q] num_items, [Q] total_num_vectors,
          flat float32 data, [Q] len(offsets), offsets (u64)
    Offsets measured in FLOATS (vectors * dim), matching your reader.
    """

    def __init__(self, out_path: str, dim: int, num_items: int):
        self.out_path = out_path
        self.dim = int(dim)
        self.num_items = int(num_items)
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp_path = self._tmp.name
        self._offsets = [0]
        self._total_floats = 0
        self._closed = False

    def append(self, item_np: np.ndarray):
        assert item_np.dtype == np.float32 and item_np.ndim == 2 and item_np.shape[1] == self.dim
        self._tmp.write(item_np.tobytes())
        self._total_floats += item_np.size
        self._offsets.append(self._total_floats)

    def close(self):
        if self._closed:
            return
        self._tmp.flush()
        self._tmp.close()
        total_num_vectors = self._total_floats // self.dim
        with open(self.out_path, "wb") as f_out, open(self._tmp_path, "rb") as f_tmp:
            f_out.write(struct.pack("Q", self.dim))
            f_out.write(struct.pack("Q", self.num_items))
            f_out.write(struct.pack("Q", total_num_vectors))
            shutil.copyfileobj(f_tmp, f_out)
            f_out.write(struct.pack("Q", len(self._offsets)))
            f_out.write(np.asarray(self._offsets, dtype=np.uint64).tobytes())
        os.remove(self._tmp_path)
        self._closed = True

    def __del__(self):
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass


# --------------------------- Helpers ---------------------------


def save_ids(ids: List[str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write("\n".join(map(str, ids)) + "\n")
    logging.info(f"Wrote IDs: {out_path} ({len(ids)})")


def _as_list(emb) -> List[torch.Tensor]:
    # Normalize ColPali outputs to list[Tensor[num_patches, dim]]
    if isinstance(emb, list):
        return [e for e in emb]
    if torch.is_tensor(emb) and emb.dim() == 3:
        return [emb[i] for i in range(emb.size(0))]
    raise ValueError("Unexpected ColPali embeddings container")


def _to_unit_norm_cpu_np(t: torch.Tensor) -> np.ndarray:
    # Row-wise L2 norm so dot == cosine downstream; returns float32 on CPU
    x = t.detach().to(dtype=torch.float32, device="cpu")
    norms = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
    x = x / norms
    return x.numpy()


# --------------------------- Encoding (streaming) ---------------------------


def encode_images_streaming(
    images: List, ids: List[str], model, processor, out_path: str, batch_size: int
):
    # Peek to get dim, then stream
    first_inputs = processor(images=images[: min(len(images), batch_size)], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        first_emb = model(**first_inputs).embeddings
    first_list = _as_list(first_emb)
    dim = int(first_list[0].shape[1])
    writer = StreamingBinaryWriter(out_path, dim=dim, num_items=len(images))

    for t in first_list:
        writer.append(_to_unit_norm_cpu_np(t))

    start = len(first_list)
    for i in range(start, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs).embeddings
        for t in _as_list(out):
            writer.append(_to_unit_norm_cpu_np(t))
        if ((i // batch_size) + 1) % 10 == 0:
            total_batches = (len(images) + batch_size - 1) // batch_size
            logging.info(f"  - Image batch {i // batch_size + 1}/{total_batches}")

    writer.close()
    save_ids(ids, os.path.splitext(out_path)[0] + ".ids.txt")


def encode_queries_streaming(
    texts: List[str], qids: List[str], model, processor, out_path: str, batch_size: int
):
    first_inputs = processor(text=texts[: min(len(texts), batch_size)], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        first_emb = model(**first_inputs).embeddings
    first_list = _as_list(first_emb)
    dim = int(first_list[0].shape[1])
    writer = StreamingBinaryWriter(out_path, dim=dim, num_items=len(texts))

    for t in first_list:
        writer.append(_to_unit_norm_cpu_np(t))

    start = len(first_list)
    for i in range(start, len(texts), batch_size):
        batch_txt = texts[i : i + batch_size]
        inputs = processor(text=batch_txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs).embeddings
        for t in _as_list(out):
            writer.append(_to_unit_norm_cpu_np(t))
        if ((i // batch_size) + 1) % 10 == 0:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logging.info(f"  - Query batch {i // batch_size + 1}/{total_batches}")

    writer.close()
    save_ids(qids, os.path.splitext(out_path)[0] + ".ids.txt")


# --------------------------- HF loader & main ---------------------------


def _pick_query_field(ds_queries):
    # Robustly pick the text field for queries
    feats = set(ds_queries.features.keys())
    for cand in ["query", "text", "question", "query_text"]:
        if cand in feats:
            return cand
    raise KeyError(f"No text field found in queries split. Available: {sorted(feats)}")


def load_vidore(dataset_name: str, split: str, debug_n: int = None):
    ds_corpus = load_dataset(
        dataset_name, "corpus", split=split
    )  # image, corpus-id (sometimes doc-id too)
    ds_queries = load_dataset(dataset_name, "queries", split=split)  # query-id + {query|text|...}

    q_text_key = _pick_query_field(ds_queries)

    images = [row["image"] for row in ds_corpus]
    img_ids = [str(row["corpus-id"]) for row in ds_corpus]

    queries = [row[q_text_key] for row in ds_queries]
    qids = [str(row["query-id"]) for row in ds_queries]

    if debug_n:
        images, img_ids = images[:debug_n], img_ids[:debug_n]
        queries, qids = queries[:debug_n], qids[:debug_n]

    # Optional visibility in logs
    import logging

    logging.info(f"Queries text field picked: '{q_text_key}'")

    return (images, img_ids), (queries, qids)


def main(args):
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
        logging.info(f"Using torch.set_num_threads({args.num_threads})")

    os.makedirs(args.output_path, exist_ok=True)

    # bf16 suggested in model card examples
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    logging.info(f"Loading model {args.model_name} with dtype={args.dtype}")
    model = ColPaliForRetrieval.from_pretrained(
        args.model_name, dtype=torch_dtype, device_map="auto"
    ).eval()
    processor = ColPaliProcessor.from_pretrained(args.model_name)

    (images, img_ids), (queries, qids) = load_vidore(
        args.dataset_name, args.split, args.debug_subset_size
    )
    logging.info(
        f"Loaded {len(images)} pages and {len(queries)} queries from {args.dataset_name}:{args.split}"
    )

    corpus_out = os.path.join(args.output_path, f"corpus_embeddings_{args.split}.bin")
    queries_out = os.path.join(args.output_path, f"query_embeddings_{args.split}.bin")

    logging.info(f"Encoding corpus (streaming, L2-normalized) -> {corpus_out}")
    encode_images_streaming(
        images, img_ids, model, processor, corpus_out, batch_size=args.image_batch_size
    )

    logging.info(f"Encoding queries (streaming, L2-normalized) -> {queries_out}")
    encode_queries_streaming(
        queries, qids, model, processor, queries_out, batch_size=args.query_batch_size
    )

    logging.info("All embedding generation complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ColPali embeddings for ViDoRe datasets (HF) with streaming + cosine-ready vectors."
    )
    ap.add_argument(
        "--dataset_name",
        type=str,
        default="vidore/economics_reports_eng_v2",
        help="ViDoRe dataset on Hugging Face.",
    )
    ap.add_argument("--split", type=str, default="test", help="Split to process.")
    ap.add_argument(
        "--output_path", type=str, required=True, help="Output dir for .bin and .ids.txt"
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="vidore/colpali-v1.3-hf",
        help="ColPali checkpoint (PaliGemma-3B).",
    )
    ap.add_argument("--image_batch_size", type=int, default=8)
    ap.add_argument("--query_batch_size", type=int, default=64)
    ap.add_argument("--debug_subset_size", type=int, default=None)
    ap.add_argument("--num_threads", type=int, default=None)
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="bf16 recommended in the model card examples.",
    )
    args = ap.parse_args()
    main(args)
