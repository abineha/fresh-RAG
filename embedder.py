"""
Mediterranean Cuisine -- RAG Embedding & Indexing Pipeline
==========================================================
Deliverable 2: Vectorisation / Embedding

Embeds chunked text using sentence-transformers models,
builds FAISS indices for semantic search, and runs a
sanity check.

Supports three embedding models:
  A) all-MiniLM-L6-v2      -- lightweight baseline (384d)
  B) all-mpnet-base-v2      -- best quality (768d)
  C) BAAI/bge-small-en-v1.5 -- retrieval specialist (384d)

Usage:
    python embedder.py                          # default: MiniLM + section_based
    python embedder.py --model all-mpnet-base-v2 --chunks chunks_fixed_200.json
    python embedder.py --run-all                # run all 6 experiment combinations
    python embedder.py --sanity-check           # quick retrieval test
"""

import os
import json
import time
import argparse
import numpy as np

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MODELS = {
    "minilm":  "all-MiniLM-L6-v2",
    "mpnet":   "all-mpnet-base-v2",
    "bge":     "BAAI/bge-small-en-v1.5",
}

# BGE requires a query prefix for retrieval (not for documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_FILES = {
    "section_based":  "chunks.json",
    "fixed_200":      "chunks_fixed_200.json",
    "fixed_500":      "chunks_fixed_500.json",
    "sentence_based": "chunks_sentence.json",
}

OUTPUT_DIR = "indices"

# The 6-combination experiment matrix
EXPERIMENT_MATRIX = [
    ("minilm", "section_based"),    # lightweight baseline
    ("mpnet",  "section_based"),    # best model + best chunks
    ("bge",    "section_based"),    # retrieval specialist
    ("mpnet",  "fixed_200"),        # small naive chunks + good model
    ("mpnet",  "fixed_500"),        # does truncation hurt?
    ("mpnet",  "sentence_based"),   # sentence boundary comparison
]


# ---------------------------------------------------------------
# STEP 1: LOAD CHUNKS
# ---------------------------------------------------------------

def load_chunks(chunk_file: str) -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks from {chunk_file}")
    return chunks


# ---------------------------------------------------------------
# STEP 2: EMBED CHUNKS
# ---------------------------------------------------------------

def embed_chunks(chunks: list[dict], model_key: str) -> dict:
    """Embed all chunks using the specified model.

    Returns dict with:
      - embeddings: numpy array (num_chunks, dim)
      - chunk_ids: list of chunk_id strings
      - model_name: full model name
      - dimension: embedding dimension
      - embed_time_s: time taken
      - truncated_count: chunks exceeding model's max token limit
      - max_seq_length: model's max sequence length
    """
    from sentence_transformers import SentenceTransformer

    model_name = MODELS[model_key]
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    max_seq = model.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    print(f"  Max sequence length: {max_seq} tokens, Dimension: {dim}")

    # Extract texts and IDs
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Check how many chunks exceed the token limit
    tokenizer = model.tokenizer
    truncated = 0
    for t in texts:
        token_count = len(tokenizer.encode(t, add_special_tokens=True))
        if token_count > max_seq:
            truncated += 1

    if truncated > 0:
        print(f"  WARNING: {truncated}/{len(texts)} chunks exceed max token limit ({max_seq})")
    else:
        print(f"  All {len(texts)} chunks fit within token limit")

    # Embed
    print(f"  Embedding {len(texts)} chunks...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise for cosine similarity
    )
    elapsed = time.time() - start
    print(f"  Embedded in {elapsed:.1f}s")

    return {
        "embeddings": embeddings,
        "chunk_ids": chunk_ids,
        "model_name": model_name,
        "model_key": model_key,
        "dimension": dim,
        "embed_time_s": round(elapsed, 2),
        "truncated_count": truncated,
        "max_seq_length": max_seq,
        "num_chunks": len(chunks),
    }


# ---------------------------------------------------------------
# STEP 3: BUILD FAISS INDEX
# ---------------------------------------------------------------

def build_faiss_index(embed_result: dict, strategy: str, output_dir: str = OUTPUT_DIR):
    """Build and save a FAISS index + chunk-ID mapping.

    Uses IndexFlatIP (inner product = cosine similarity on normalised vectors).
    """
    import faiss

    os.makedirs(output_dir, exist_ok=True)

    embeddings = embed_result["embeddings"]
    dim = embed_result["dimension"]
    model_key = embed_result["model_key"]

    # Create flat inner-product index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  FAISS index: {index.ntotal} vectors, {dim}d")

    # File names
    prefix = f"{model_key}_{strategy}"
    index_path = os.path.join(output_dir, f"faiss_{prefix}.bin")
    mapping_path = os.path.join(output_dir, f"mapping_{prefix}.json")
    meta_path = os.path.join(output_dir, f"meta_{prefix}.json")

    # Save index
    faiss.write_index(index, index_path)
    print(f"  Saved index:   {index_path}")

    # Save chunk-ID mapping (position -> chunk_id)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(embed_result["chunk_ids"], f, indent=2)
    print(f"  Saved mapping: {mapping_path}")

    # Save metadata
    meta = {
        "model_name": embed_result["model_name"],
        "model_key": model_key,
        "strategy": strategy,
        "dimension": dim,
        "num_chunks": embed_result["num_chunks"],
        "embed_time_s": embed_result["embed_time_s"],
        "truncated_count": embed_result["truncated_count"],
        "max_seq_length": embed_result["max_seq_length"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved meta:    {meta_path}")

    return index_path, mapping_path, meta_path


# ---------------------------------------------------------------
# STEP 4: SANITY CHECK
# ---------------------------------------------------------------

def sanity_check(model_key: str = "minilm", strategy: str = "section_based",
                 top_k: int = 5):
    """Embed test queries and retrieve top-k chunks to verify the pipeline."""
    import faiss
    from sentence_transformers import SentenceTransformer

    prefix = f"{model_key}_{strategy}"
    index_path = os.path.join(OUTPUT_DIR, f"faiss_{prefix}.bin")
    mapping_path = os.path.join(OUTPUT_DIR, f"mapping_{prefix}.json")

    if not os.path.exists(index_path):
        print(f"  Index not found: {index_path}. Run embedding first.")
        return

    # Load index and mapping
    index = faiss.read_index(index_path)
    with open(mapping_path, encoding="utf-8") as f:
        chunk_ids = json.load(f)

    # Load chunks for text lookup
    chunk_file = CHUNK_FILES[strategy]
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)
    id_to_chunk = {c["chunk_id"]: c for c in chunks}

    # Load model
    model_name = MODELS[model_key]
    model = SentenceTransformer(model_name)

    test_queries = [
        "What are the main ingredients of hummus?",
        "What is paella and where does it come from?",
        "What distinguishes Moroccan tagine from Tunisian tagine?",
    ]

    print(f"\n  Sanity check: {model_name} + {strategy}")
    print(f"  {'-'*60}")

    for query in test_queries:
        # Add BGE prefix if needed
        encode_query = query
        if model_key == "bge":
            encode_query = BGE_QUERY_PREFIX + query

        q_vec = model.encode([encode_query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_vec, top_k)

        print(f"\n  Q: {query}")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            cid = chunk_ids[idx]
            chunk = id_to_chunk.get(cid, {})
            title = chunk.get("doc_title", "?")
            section = chunk.get("section", "?")
            wc = chunk.get("word_count", 0)
            print(f"    {rank}. [{score:.4f}] {cid}  ({title} / {section}, {wc}w)")


# ---------------------------------------------------------------
# STEP 5: RUN ONE COMBINATION
# ---------------------------------------------------------------

def run_one(model_key: str, strategy: str):
    """Run embedding + indexing for one model-strategy combination."""
    chunk_file = CHUNK_FILES[strategy]
    if not os.path.exists(chunk_file):
        print(f"  Chunk file not found: {chunk_file}. Run chunker.py first.")
        return

    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} + {strategy}")
    print(f"{'='*60}")

    chunks = load_chunks(chunk_file)
    result = embed_chunks(chunks, model_key)
    build_faiss_index(result, strategy)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed chunks and build FAISS indices"
    )
    parser.add_argument("--model", default="minilm",
                        choices=list(MODELS.keys()),
                        help="Embedding model to use")
    parser.add_argument("--chunks", default=None,
                        help="Chunk JSON file (overrides --strategy)")
    parser.add_argument("--strategy", default="section_based",
                        choices=list(CHUNK_FILES.keys()),
                        help="Chunking strategy (determines chunk file)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 6 experiment combinations")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity check queries after embedding")
    args = parser.parse_args()

    if args.run_all:
        for model_key, strategy in EXPERIMENT_MATRIX:
            run_one(model_key, strategy)
        # Sanity check on the primary combination
        print("\n" + "="*60)
        print("  SANITY CHECK")
        print("="*60)
        sanity_check("mpnet", "section_based")
    else:
        if args.chunks:
            # Custom chunk file -- infer strategy from filename
            strategy = args.chunks.replace("chunks_", "").replace(".json", "")
            CHUNK_FILES[strategy] = args.chunks
        else:
            strategy = args.strategy

        run_one(args.model, strategy)

        if args.sanity_check:
            sanity_check(args.model, strategy)
