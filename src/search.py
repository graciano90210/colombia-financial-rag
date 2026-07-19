import json
import numpy as np
from pathlib import Path
from onnxruntime import InferenceSession
from tokenizers import Tokenizer
import minsearch


MODELS_PATH = Path(__file__).parent.parent / "models"
MODEL_DIR = MODELS_PATH / "Xenova" / "all-MiniLM-L6-v2"


# ─── Embedding ────────────────────────────────────────────────────────────────

class Embedder:
    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        self.session = InferenceSession(str(model_dir / "model.onnx"))

    def _mean_pooling(self, output, attention_mask):
        mask = np.expand_dims(attention_mask, -1).astype(np.float32)
        return (output * mask).sum(axis=1) / mask.sum(axis=1)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type = np.zeros_like(ids)
        output = self.session.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type,
        })[0]
        vecs = self._mean_pooling(output, mask)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


# ─── Índices ─────────────────────────────────────────────────────────────────

class SearchEngine:
    def __init__(self, chunks, embedder=None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()
        self._build_keyword_index()
        self._build_vector_index()

    def _build_keyword_index(self):
        self.kw_index = minsearch.Index(
            text_fields=["text"],
            keyword_fields=["company"],
        )
        self.kw_index.fit(self.chunks)

    def _build_vector_index(self, batch_size=32):
        texts = [c["text"] for c in self.chunks]
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            all_vecs.append(self.embedder.embed(batch))
            print(f"  Embeddings: {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        self.vectors = np.vstack(all_vecs)

    # ── Keyword ──────────────────────────────────────────────────────────────

    def keyword_search(self, query, company=None, top_k=5):
        filter_dict = {"company": company} if company else {}
        return self.kw_index.search(
            query,
            filter_dict=filter_dict,
            num_results=top_k,
        )

    # ── Vector ───────────────────────────────────────────────────────────────

    def vector_search(self, query, company=None, top_k=5):
        q_vec = self.embedder.embed(query)[0]
        scores = self.vectors @ q_vec

        if company:
            mask = np.array([c["company"] == company for c in self.chunks])
            scores = np.where(mask, scores, -1.0)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]

    # ── Hybrid (RRF) ─────────────────────────────────────────────────────────

    def hybrid_search(self, query, company=None, top_k=5, rrf_k=60):
        kw_results = self.keyword_search(query, company=company, top_k=top_k * 2)
        vec_results = self.vector_search(query, company=company, top_k=top_k * 2)

        # Construir ranking por texto (identificador único del chunk)
        def chunk_key(c):
            return (c["company"], c["page_start"])

        kw_ranks = {chunk_key(c): i for i, c in enumerate(kw_results)}
        vec_ranks = {chunk_key(c): i for i, c in enumerate(vec_results)}

        all_keys = set(kw_ranks) | set(vec_ranks)
        rrf_scores = {}
        for key in all_keys:
            score = 0.0
            if key in kw_ranks:
                score += 1.0 / (rrf_k + kw_ranks[key] + 1)
            if key in vec_ranks:
                score += 1.0 / (rrf_k + vec_ranks[key] + 1)
            rrf_scores[key] = score

        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # Reconstruir lista de chunks preservando orden RRF
        key_to_chunk = {chunk_key(c): c for c in kw_results + vec_results}
        return [key_to_chunk[k] for k in sorted_keys if k in key_to_chunk]


# ─── Carga rápida ────────────────────────────────────────────────────────────

def load_engine(chunks_path="data/chunks.json"):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Cargando motor de búsqueda con {len(chunks)} chunks...")
    return SearchEngine(chunks)


if __name__ == "__main__":
    engine = load_engine()
    query = "¿Cuál fue la utilidad neta de Bancolombia en 2024?"
    print(f"\nQuery: {query}\n")
    print("=== Keyword Search ===")
    for c in engine.keyword_search(query, company="Bancolombia"):
        print(f"  [{c['company']} p{c['page_start']}-{c['page_end']}] {c['text'][:120]}...")
    print("\n=== Vector Search ===")
    for c in engine.vector_search(query, company="Bancolombia"):
        print(f"  [{c['company']} p{c['page_start']}-{c['page_end']}] {c['text'][:120]}...")
    print("\n=== Hybrid Search (RRF) ===")
    for c in engine.hybrid_search(query, company="Bancolombia"):
        print(f"  [{c['company']} p{c['page_start']}-{c['page_end']}] {c['text'][:120]}...")
