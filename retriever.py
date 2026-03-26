"""
Mediterranean Cuisine -- RAG Retrieval & Ranking Pipeline
=========================================================
Deliverable 3: Retrieval and Ranking

Three retrieval strategies:
  A) Pure vector retrieval  (cosine similarity via FAISS)
  B) Pure BM25 keyword retrieval
  C) Hybrid retrieval with Reciprocal Rank Fusion (RRF)

Usage:
    python retriever.py                        # default: vector + mpnet + section_based
    python retriever.py --method bm25          # BM25 only
    python retriever.py --method hybrid        # hybrid RRF fusion
    python retriever.py --run-all              # run full experiment matrix
    python retriever.py --interactive          # interactive query mode
"""

import os
import json
import argparse
import numpy as np

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet":  "all-mpnet-base-v2",
    "bge":    "BAAI/bge-small-en-v1.5",
}

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_FILES = {
    "section_based":  "chunks.json",
    "fixed_200":      "chunks_fixed_200.json",
    "fixed_500":      "chunks_fixed_500.json",
    "sentence_based": "chunks_sentence.json",
}

INDEX_DIR = "indices"
RESULTS_DIR = "retrieval_results"

BENCHMARK_QUERIES_FILE = "rag_benchmark_queries.json"

# Experiment matrix: (method, model_key, strategy)
EXPERIMENT_MATRIX = [
    ("vector",  "mpnet", "section_based"),   # best embedding setup
    ("bm25",    None,    "section_based"),    # keyword-only baseline
    ("hybrid",  "mpnet", "section_based"),    # does fusion beat either alone?
    ("vector",  "mpnet", "fixed_200"),        # best truncation-free setup
    ("hybrid",  "mpnet", "fixed_200"),        # fusion on small chunks
    ("vector",  "bge",   "section_based"),    # retrieval-specialist model
]


# ---------------------------------------------------------------
# STEP 1: LOAD CHUNKS
# ---------------------------------------------------------------

def load_chunks(chunk_file: str) -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


# ---------------------------------------------------------------
# STEP 2: SETUP -- load indices and models
# ---------------------------------------------------------------

def setup_vector(model_key: str, strategy: str):
    """Load FAISS index, ID mapping, and embedding model for vector retrieval."""
    import faiss
    from sentence_transformers import SentenceTransformer

    prefix = f"{model_key}_{strategy}"
    index_path = os.path.join(INDEX_DIR, f"faiss_{prefix}.bin")
    mapping_path = os.path.join(INDEX_DIR, f"mapping_{prefix}.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    with open(mapping_path, encoding="utf-8") as f:
        id_mapping = json.load(f)

    model_name = MODELS[model_key]
    model = SentenceTransformer(model_name)

    print(f"  Loaded FAISS index: {index.ntotal} vectors ({index_path})")
    print(f"  Loaded model: {model_name}")

    return {
        "index": index,
        "id_mapping": id_mapping,
        "model": model,
        "model_key": model_key,
    }


def setup_bm25(strategy: str):
    """Build a BM25 index from chunks."""
    from rank_bm25 import BM25Okapi

    chunk_file = CHUNK_FILES[strategy]
    chunks = load_chunks(chunk_file)

    # Tokenize: lowercase + whitespace split
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    id_mapping = [c["chunk_id"] for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  Built BM25 index: {len(chunks)} documents ({chunk_file})")

    return {
        "bm25": bm25,
        "id_mapping": id_mapping,
    }


# ---------------------------------------------------------------
# STEP 3: RETRIEVAL FUNCTIONS
# ---------------------------------------------------------------

def retrieve_vector(query: str, vec_setup: dict, k: int = 5) -> list[dict]:
    """Embed query and search FAISS index. Returns top-k results."""
    model = vec_setup["model"]
    index = vec_setup["index"]
    id_mapping = vec_setup["id_mapping"]
    model_key = vec_setup["model_key"]

    # BGE needs query prefix
    encode_query = query
    if model_key == "bge":
        encode_query = BGE_QUERY_PREFIX + query

    q_vec = model.encode([encode_query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        results.append({
            "chunk_id": id_mapping[idx],
            "rank": rank,
            "score": round(float(score), 6),
        })
    return results


def retrieve_bm25(query: str, bm25_setup: dict, k: int = 5) -> list[dict]:
    """Score query with BM25 and return top-k results."""
    bm25 = bm25_setup["bm25"]
    id_mapping = bm25_setup["id_mapping"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "chunk_id": id_mapping[idx],
            "rank": rank,
            "score": round(float(scores[idx]), 6),
        })
    return results


def retrieve_hybrid(query: str, vec_setup: dict, bm25_setup: dict,
                    k: int = 5, rrf_k: int = 60, fetch_k: int = 20) -> list[dict]:
    """Run vector + BM25, fuse with Reciprocal Rank Fusion, return top-k."""
    vec_results = retrieve_vector(query, vec_setup, k=fetch_k)
    bm25_results = retrieve_bm25(query, bm25_setup, k=fetch_k)

    # Compute RRF scores
    rrf_scores = {}

    for r in vec_results:
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rrf_k + r["rank"])

    for r in bm25_results:
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rrf_k + r["rank"])

    # Sort by RRF score descending
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (cid, score) in enumerate(sorted_chunks[:k], 1):
        results.append({
            "chunk_id": cid,
            "rank": rank,
            "score": round(score, 6),
        })
    return results


# ---------------------------------------------------------------
# STEP 4: RUN RETRIEVAL ON BENCHMARK QUERIES
# ---------------------------------------------------------------

def run_benchmark(method: str, model_key: str, strategy: str,
                  k: int = 5) -> list[dict]:
    """Run all 15 benchmark queries with the specified retrieval config."""
    # Load benchmark queries
    with open(BENCHMARK_QUERIES_FILE, encoding="utf-8") as f:
        benchmark = json.load(f)
    queries = benchmark["queries"]

    # Setup indices
    vec_setup = None
    bm25_setup = None

    if method in ("vector", "hybrid"):
        vec_setup = setup_vector(model_key, strategy)
    if method in ("bm25", "hybrid"):
        bm25_setup = setup_bm25(strategy)

    # Retrieve for each query
    all_results = []
    print(f"\n  Running {len(queries)} benchmark queries...")

    for q in queries:
        query = q["query"]
        query_id = q["query_id"]

        if method == "vector":
            retrieved = retrieve_vector(query, vec_setup, k=k)
        elif method == "bm25":
            retrieved = retrieve_bm25(query, bm25_setup, k=k)
        elif method == "hybrid":
            retrieved = retrieve_hybrid(query, vec_setup, bm25_setup, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")

        result = {
            "query_id": query_id,
            "query": query,
            "method": method if method != "hybrid" else "hybrid_rrf",
            "model": MODELS.get(model_key, "none") if model_key else "none",
            "chunking": strategy,
            "retrieved": retrieved,
        }
        all_results.append(result)

    return all_results


def print_results(all_results: list[dict], chunks_lookup: dict):
    """Print retrieval results with chunk previews."""
    for r in all_results:
        print(f"\n  Q{r['query_id']}: {r['query']}")
        for hit in r["retrieved"]:
            cid = hit["chunk_id"]
            chunk = chunks_lookup.get(cid, {})
            title = chunk.get("doc_title", "?")
            section = chunk.get("section", "?")
            wc = chunk.get("word_count", 0)
            print(f"    {hit['rank']}. [{hit['score']:.4f}] {cid}  ({title} / {section}, {wc}w)")


def save_results(all_results: list[dict], method: str, model_key: str,
                 strategy: str):
    """Save retrieval results to JSON for later evaluation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_tag = model_key if model_key else "none"
    filename = f"retrieval_{method}_{model_tag}_{strategy}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved results: {filepath}")

    return filepath


# ---------------------------------------------------------------
# STEP 5: RUN ONE EXPERIMENT
# ---------------------------------------------------------------

def run_one(method: str, model_key: str, strategy: str, k: int = 5):
    """Run one retrieval experiment and save results."""
    model_label = MODELS.get(model_key, "none") if model_key else "none"
    print(f"\n{'='*60}")
    print(f"  {method.upper()} | {model_label} | {strategy}")
    print(f"{'='*60}")

    results = run_benchmark(method, model_key, strategy, k=k)

    # Load chunks for display
    chunk_file = CHUNK_FILES[strategy]
    chunks = load_chunks(chunk_file)
    chunks_lookup = {c["chunk_id"]: c for c in chunks}

    print_results(results, chunks_lookup)
    save_results(results, method, model_key, strategy)

    return results


# ---------------------------------------------------------------
# STEP 6: INTERACTIVE MODE
# ---------------------------------------------------------------

def interactive_mode(method: str, model_key: str, strategy: str, k: int = 5):
    """Interactive query loop for testing retrieval."""
    print(f"\n  Interactive mode: {method} | {MODELS.get(model_key, 'none')} | {strategy}")
    print(f"  Type 'quit' to exit.\n")

    # Setup
    vec_setup = None
    bm25_setup = None

    if method in ("vector", "hybrid"):
        vec_setup = setup_vector(model_key, strategy)
    if method in ("bm25", "hybrid"):
        bm25_setup = setup_bm25(strategy)

    # Load chunks for display
    chunk_file = CHUNK_FILES[strategy]
    chunks = load_chunks(chunk_file)
    chunks_lookup = {c["chunk_id"]: c for c in chunks}

    while True:
        try:
            query = input("\n  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() == "quit":
            break

        if method == "vector":
            results = retrieve_vector(query, vec_setup, k=k)
        elif method == "bm25":
            results = retrieve_bm25(query, bm25_setup, k=k)
        elif method == "hybrid":
            results = retrieve_hybrid(query, vec_setup, bm25_setup, k=k)

        for hit in results:
            cid = hit["chunk_id"]
            chunk = chunks_lookup.get(cid, {})
            title = chunk.get("doc_title", "?")
            section = chunk.get("section", "?")
            wc = chunk.get("word_count", 0)
            preview = chunk.get("text", "")[:120].encode("ascii", "replace").decode()
            print(f"    {hit['rank']}. [{hit['score']:.4f}] {cid}  ({title} / {section}, {wc}w)")
            print(f"       {preview}...")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve chunks using vector, BM25, or hybrid methods"
    )
    parser.add_argument("--method", default="vector",
                        choices=["vector", "bm25", "hybrid"],
                        help="Retrieval method")
    parser.add_argument("--model", default="mpnet",
                        choices=list(MODELS.keys()),
                        help="Embedding model (ignored for BM25)")
    parser.add_argument("--strategy", default="section_based",
                        choices=list(CHUNK_FILES.keys()),
                        help="Chunking strategy")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results to retrieve")
    parser.add_argument("--run-all", action="store_true",
                        help="Run full experiment matrix")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive query mode")
    args = parser.parse_args()

    if args.run_all:
        for method, model_key, strategy in EXPERIMENT_MATRIX:
            run_one(method, model_key, strategy, k=args.k)
        print(f"\n{'='*60}")
        print(f"  ALL EXPERIMENTS COMPLETE")
        print(f"  Results saved to {RESULTS_DIR}/")
        print(f"{'='*60}")
    elif args.interactive:
        model_key = args.model if args.method != "bm25" else None
        interactive_mode(args.method, model_key, args.strategy, k=args.k)
    else:
        model_key = args.model if args.method != "bm25" else None
        run_one(args.method, model_key, args.strategy, k=args.k)
