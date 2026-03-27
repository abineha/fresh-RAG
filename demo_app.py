"""
Mediterranean Cuisine RAG -- Streamlit Demo Application
========================================================
Interactive demo for the complete RAG pipeline:
  1. Configure retrieval method, embedding model, chunking strategy
  2. Ask questions and see retrieved chunks + generated answers
  3. Run evaluation against gold-standard benchmark

Usage:
    streamlit run demo_app.py
"""

import os
import json
import time
import streamlit as st
import numpy as np

# ---------------------------------------------------------------
# IMPORT PIPELINE MODULES
# ---------------------------------------------------------------
from retriever import (
    MODELS, CHUNK_FILES, BGE_QUERY_PREFIX, BGEM3_QUERY_PREFIX,
    load_chunks as retriever_load_chunks,
    retrieve_vector, retrieve_bm25, retrieve_hybrid,
)
from generator import (
    MODEL_NAME as LLM_MODEL_NAME,
    SYSTEM_PROMPTS, GENERATION_CONFIG,
    load_model, format_context, build_messages, generate_answer,
)
from evaluator import (
    load_gold_standard, evaluate_retrieval, chunks_match,
    compute_rouge_l, compute_bert_score, compute_faithfulness,
)

# ---------------------------------------------------------------
# MODEL KEY MAPPING
# ---------------------------------------------------------------
MODEL_NAME_TO_KEY = {v: k for k, v in MODELS.items()}


# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Mediterranean Cuisine RAG",
    page_icon="🫒",
    layout="wide",
)

# ---------------------------------------------------------------
# CUSTOM THEME CSS
# Palette: #E27396 #EA9AB2 #EFCFE3 #EAF2D7 #B3DEE2
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #EFCFE3;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #333333;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #EAF2D7;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #333333;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E27396 !important;
        color: white !important;
        border-radius: 6px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #B3DEE2;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #E27396;
    }
    div[data-testid="stMetric"] label {
        color: #555555;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #333333;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        background-color: #EAF2D7;
        border-radius: 6px;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #E27396;
        border-color: #E27396;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #EA9AB2;
        border-color: #EA9AB2;
        color: white;
    }

    /* Info/success boxes */
    div[data-testid="stAlert"] {
        border-left-color: #E27396;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E27396 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #B3DEE2;
        border-radius: 8px;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #E27396;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------------
st.sidebar.title("RAG Configuration")

retrieval_method = st.sidebar.selectbox(
    "Retrieval Method",
    ["vector", "bm25", "hybrid"],
    index=0,  # default: vector (best MRR=1.0, R@5=0.98)
    help="Vector (FAISS), BM25 (keyword), or Hybrid RRF fusion",
)

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    list(MODELS.values()),
    index=1,  # default: all-mpnet-base-v2 (best overall)
    disabled=(retrieval_method == "bm25"),
    help="Embedding model for vector search (ignored for BM25)",
)

chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    list(CHUNK_FILES.keys()),
    index=0,  # default: section_based (best recall=0.98)
    help="How documents were split into chunks",
)

prompt_strategy = st.sidebar.selectbox(
    "Prompt Strategy",
    ["zero_shot", "few_shot", "structured"],
    index=2,  # default: structured (best ROUGE-L=0.25, BERTScore=0.82)
    help="How the prompt is constructed for the LLM",
)

top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 5)

gold_file = st.sidebar.text_input(
    "Gold Standard File",
    value="rag_benchmark_answers.json",
    help="Path to gold-standard answers for evaluation",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Defaults = best config from evaluation:**  \n"
    "Vector + mpnet + section_based (MRR 1.0)  \n"
    "Structured prompt (ROUGE-L 0.25)",
    help="Based on evaluation across 10 retrieval and 3 generation experiments",
)
st.sidebar.caption(f"LLM: {LLM_MODEL_NAME}")
st.sidebar.caption(f"Generation: temp={GENERATION_CONFIG['temperature']}, "
                    f"top_p={GENERATION_CONFIG['top_p']}")


# ---------------------------------------------------------------
# FUNCTION 1: LOAD PIPELINE RESOURCES (cached)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline resources...")
def load_pipeline_resources(retrieval_method, embedding_model, chunking_strategy):
    """Load all resources needed for the pipeline. Cached by Streamlit."""
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    resources = {}

    # 1. Load chunks
    chunk_file = CHUNK_FILES[chunking_strategy]
    chunks = retriever_load_chunks(chunk_file)
    resources["chunks"] = chunks
    resources["chunks_lookup"] = {c["chunk_id"]: c for c in chunks}

    # 2. Load embedding model + FAISS index (only for vector/hybrid)
    model_key = MODEL_NAME_TO_KEY.get(embedding_model)

    if retrieval_method in ("vector", "hybrid") and model_key:
        # Load FAISS index
        prefix = f"{model_key}_{chunking_strategy}"
        index_path = os.path.join("indices", f"faiss_{prefix}.bin")
        mapping_path = os.path.join("indices", f"mapping_{prefix}.json")

        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            with open(mapping_path, encoding="utf-8") as f:
                id_mapping = json.load(f)

            # Load embedding model
            embed_model = SentenceTransformer(embedding_model)

            resources["vec_setup"] = {
                "index": index,
                "id_mapping": id_mapping,
                "model": embed_model,
                "model_key": model_key,
            }

    # 3. Build BM25 index (only for bm25/hybrid)
    if retrieval_method in ("bm25", "hybrid"):
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        bm25_id_mapping = [c["chunk_id"] for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        resources["bm25_setup"] = {
            "bm25": bm25,
            "id_mapping": bm25_id_mapping,
        }

    # 4. Load Qwen model + tokenizer
    llm_model, tokenizer = load_model()
    resources["llm_model"] = llm_model
    resources["tokenizer"] = tokenizer

    return resources


# ---------------------------------------------------------------
# FUNCTION 2: RUN SINGLE QUERY
# ---------------------------------------------------------------
def run_single_query(query, config, resources, prompt_strategy):
    """Run the full RAG pipeline for a single query.

    Returns (response, retrieved_context, elapsed_time).
    """
    method = config["retrieval_method"]
    k = config["top_k"]

    # Step 1: Retrieve
    start = time.time()

    if method == "vector":
        results = retrieve_vector(query, resources["vec_setup"], k=k)
    elif method == "bm25":
        results = retrieve_bm25(query, resources["bm25_setup"], k=k)
    elif method == "hybrid":
        results = retrieve_hybrid(
            query, resources["vec_setup"], resources["bm25_setup"], k=k
        )

    retrieval_time = time.time() - start

    # Step 2: Resolve chunk IDs to full chunk objects
    chunks_lookup = resources["chunks_lookup"]
    retrieved_context = []
    for hit in results:
        cid = hit["chunk_id"]
        chunk = chunks_lookup.get(cid, {})
        retrieved_context.append({
            "doc_id": cid,
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "title": chunk.get("doc_title", ""),
            "score": hit["score"],
            "rank": hit["rank"],
        })

    # Step 3: Generate
    context_str = format_context(retrieved_context)
    messages = build_messages(prompt_strategy, context_str, query)

    gen_start = time.time()
    response, gen_time = generate_answer(
        resources["llm_model"], resources["tokenizer"], messages
    )
    total_time = time.time() - start

    return response, retrieved_context, {
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": round(gen_time, 2),
        "total_time": round(total_time, 2),
    }


# ---------------------------------------------------------------
# FUNCTION 3: RUN EVALUATION
# ---------------------------------------------------------------
def run_evaluation(generated_results, config, resources, gold_file_path):
    """Evaluate generated results against gold standard.

    Returns dict with retrieval and generation metrics.
    """
    # Load gold standard
    gold_standard = load_gold_standard()
    chunks_lookup = resources["chunks_lookup"]

    # Build retrieval results in the format evaluate_retrieval expects
    retrieval_formatted = []
    gold_answers = []
    generated_answers = []
    retrieved_contexts = []

    for gen in generated_results:
        qid = gen["query_id"]

        # Build retrieval format: list of {chunk_id, ...}
        retrieved = []
        for ctx in gen["retrieved_context"]:
            retrieved.append({
                "chunk_id": ctx["doc_id"],
                "rank": ctx.get("rank", 0),
                "score": ctx.get("score", 0),
            })
        retrieval_formatted.append({
            "query_id": qid,
            "retrieved": retrieved,
        })

        generated_answers.append(gen["response"])
        retrieved_contexts.append(gen["retrieved_context"])

    # Gold answers in order
    gold_answers = [g["response"] for g in gold_standard]

    # Retrieval metrics
    retr_metrics = evaluate_retrieval(retrieval_formatted, gold_standard, chunks_lookup)

    # ROUGE-L
    rouge_scores, rouge_avg = compute_rouge_l(gold_answers, generated_answers)

    # BERTScore
    bert_scores, bert_avg = compute_bert_score(gold_answers, generated_answers)

    # Faithfulness
    faith_scores, faith_avg = compute_faithfulness(generated_answers, retrieved_contexts)

    return {
        "retrieval": {
            "precision@5": retr_metrics["avg_precision_at_5"],
            "recall@5": retr_metrics["avg_recall_at_5"],
            "mrr": retr_metrics["mrr"],
        },
        "generation": {
            "rouge_l": rouge_avg,
            "bertscore_f1": bert_avg,
        },
        "faithfulness": faith_avg,
        "per_query": {
            "rouge_l": rouge_scores,
            "bertscore_f1": bert_scores,
            "faithfulness": faith_scores,
            "retrieval": retr_metrics["per_query"],
        },
    }


# ---------------------------------------------------------------
# UI: TABS
# ---------------------------------------------------------------
st.title("Mediterranean Cuisine RAG System")

tab_query, tab_benchmark, tab_eval, tab_explore, tab_about = st.tabs(
    ["Single Query", "Benchmark", "Evaluation", "Exploration Results", "About"]
)

# ---------------------------------------------------------------
# TAB 1: Single Query
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# TAB 1: Single Query + Upload JSON
# ---------------------------------------------------------------
with tab_query:
    st.header("Ask a Question")

    # -------------------------------
    # SINGLE QUERY INPUT
    # -------------------------------
    query = st.text_input(
        "Enter your question about Mediterranean cuisine:",
        placeholder="e.g., What are the main ingredients of hummus?",
    )

    if st.button("Get Answer", type="primary", disabled=not query):
        config = {
            "retrieval_method": retrieval_method,
            "top_k": top_k,
        }

        with st.spinner("Loading resources..."):
            resources = load_pipeline_resources(
                retrieval_method, embedding_model, chunking_strategy
            )

        with st.spinner("Retrieving and generating..."):
            response, context, timing = run_single_query(
                query, config, resources, prompt_strategy
            )

        st.subheader("Answer")
        st.write(response)

        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval", f"{timing['retrieval_time']}s")
        col2.metric("Generation", f"{timing['generation_time']}s")
        col3.metric("Total", f"{timing['total_time']}s")

        st.subheader("Retrieved Context")
        for ctx in context:
            with st.expander(
                f"[{ctx['rank']}] {ctx['doc_id']} (score: {ctx['score']:.4f})"
            ):
                st.caption(f"Source: {ctx.get('source', 'N/A')} | "
                           f"Title: {ctx.get('title', 'N/A')}")
                st.write(ctx["text"])

    # -------------------------------
    # JSON FILE UPLOAD
    # -------------------------------
    st.markdown("---")
    st.subheader("Or Upload a JSON File")

    uploaded_file = st.file_uploader(
        "Upload a JSON file with queries",
        type=["json"]
    )

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)

            if "queries" not in data:
                st.error("Invalid JSON format. Must contain 'queries' key.")
            else:
                queries = data["queries"]
                st.success(f"Loaded {len(queries)} queries!")

                if st.button("Run Uploaded Queries", type="primary"):
                    config = {
                        "retrieval_method": retrieval_method,
                        "top_k": top_k,
                    }

                    with st.spinner("Loading resources..."):
                        resources = load_pipeline_resources(
                            retrieval_method, embedding_model, chunking_strategy
                        )

                    all_results = []
                    progress = st.progress(0, text="Running uploaded queries...")

                    for i, q in enumerate(queries):
                        progress.progress(
                            (i + 1) / len(queries),
                            text=f"Query {i + 1}/{len(queries)}: {q['query'][:60]}..."
                        )

                        response, context, timing = run_single_query(
                            q["query"], config, resources, prompt_strategy
                        )

                        all_results.append({
                            "query_id": q.get("query_id", str(i)),
                            "query": q["query"],
                            "response": response,
                            "retrieved_context": context,
                            "timing": timing,
                        })

                    progress.empty()
                    st.success("Completed all uploaded queries!")

                    # Save for evaluation tab
                    st.session_state["benchmark_results"] = all_results

                    # Display results
                    for r in all_results:
                        with st.expander(f"Q{r['query_id']}: {r['query']}"):
                            st.write("**Answer:**", r["response"])
                            st.caption(
                                f"Retrieval: {r['timing']['retrieval_time']}s | "
                                f"Generation: {r['timing']['generation_time']}s"
                            )

                            for ctx in r["retrieved_context"]:
                                st.caption(
                                    f"[{ctx['rank']}] {ctx['doc_id']} "
                                    f"(score: {ctx['score']:.4f})"
                                )

        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")
# ---------------------------------------------------------------
# TAB 2: Benchmark (run all 15 queries)
# ---------------------------------------------------------------
with tab_benchmark:
    st.header("Run Benchmark Queries")
    st.write("Run all 15 benchmark queries and view results side by side.")

    if st.button("Run All 15 Queries", type="primary"):
        config = {
            "retrieval_method": retrieval_method,
            "top_k": top_k,
        }

        with st.spinner("Loading resources..."):
            resources = load_pipeline_resources(
                retrieval_method, embedding_model, chunking_strategy
            )

        # Load benchmark queries
        with open("rag_benchmark_queries.json", encoding="utf-8") as f:
            benchmark = json.load(f)
        queries = benchmark["queries"]

        all_results = []
        progress = st.progress(0, text="Running queries...")

        for i, q in enumerate(queries):
            progress.progress(
                (i + 1) / len(queries),
                text=f"Query {i + 1}/{len(queries)}: {q['query'][:60]}...",
            )
            response, context, timing = run_single_query(
                q["query"], config, resources, prompt_strategy
            )
            all_results.append({
                "query_id": q["query_id"],
                "query": q["query"],
                "response": response,
                "retrieved_context": context,
                "timing": timing,
            })

        progress.empty()
        st.success(f"Completed {len(all_results)} queries!")

        # Store in session state for evaluation tab
        st.session_state["benchmark_results"] = all_results

        # Display results
        for r in all_results:
            with st.expander(f"Q{r['query_id']}: {r['query']}"):
                st.write("**Answer:**", r["response"])
                st.caption(
                    f"Retrieval: {r['timing']['retrieval_time']}s | "
                    f"Generation: {r['timing']['generation_time']}s"
                )
                for ctx in r["retrieved_context"]:
                    st.caption(
                        f"  [{ctx['rank']}] {ctx['doc_id']} "
                        f"(score: {ctx['score']:.4f})"
                    )

# ---------------------------------------------------------------
# TAB 3: Evaluation
# ---------------------------------------------------------------
with tab_eval:
    st.header("Evaluation Metrics")

    has_results = "benchmark_results" in st.session_state

    if not has_results:
        st.info("Run the benchmark queries first (Benchmark tab), "
                "then come back here to evaluate.")

    if st.button("Run Evaluation", type="primary", disabled=not has_results):
        with st.spinner("Loading resources..."):
            resources = load_pipeline_resources(
                retrieval_method, embedding_model, chunking_strategy
            )

        with st.spinner("Computing metrics (ROUGE-L, BERTScore, Faithfulness)..."):
            metrics = run_evaluation(
                st.session_state["benchmark_results"],
                {"retrieval_method": retrieval_method, "top_k": top_k},
                resources,
                gold_file,
            )

        st.session_state["eval_metrics"] = metrics

    if "eval_metrics" in st.session_state:
        metrics = st.session_state["eval_metrics"]

        # Retrieval metrics
        st.subheader("Retrieval Quality")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision@5", f"{metrics['retrieval']['precision@5']:.4f}")
        col2.metric("Recall@5", f"{metrics['retrieval']['recall@5']:.4f}")
        col3.metric("MRR", f"{metrics['retrieval']['mrr']:.4f}")

        # Generation metrics
        st.subheader("Generation Quality")
        col1, col2, col3 = st.columns(3)
        col1.metric("ROUGE-L", f"{metrics['generation']['rouge_l']:.4f}")
        col2.metric("BERTScore F1", f"{metrics['generation']['bertscore_f1']:.4f}")
        col3.metric("Faithfulness", f"{metrics['faithfulness']:.4f}")

        # Per-query details
        st.subheader("Per-Query Breakdown")
        import pandas as pd

        pq = metrics["per_query"]
        df = pd.DataFrame({
            "Query": [f"Q{i}" for i in range(len(pq["rouge_l"]))],
            "ROUGE-L": pq["rouge_l"],
            "BERTScore": pq["bertscore_f1"],
            "Faithfulness": pq["faithfulness"],
            "P@5": [q["precision_at_5"] for q in pq["retrieval"]],
            "R@5": [q["recall_at_5"] for q in pq["retrieval"]],
            "RR": [q["reciprocal_rank"] for q in pq["retrieval"]],
        })
        st.dataframe(df, width="stretch")

    # Link to exploration tab for full comparison
    st.caption("See the **Exploration Results** tab for the full comparison across all 10 retrieval and 3 generation experiments.")

# ---------------------------------------------------------------
# TAB 4: Exploration Results
# ---------------------------------------------------------------
with tab_explore:
    st.header("Pipeline Exploration Results")
    st.write(
        "Summary of all configurations tested during development. "
        "The **best pipeline** is highlighted."
    )

    eval_file = os.path.join("evaluation_results", "evaluation_results.json")
    if os.path.exists(eval_file):
        import pandas as pd

        with open(eval_file, encoding="utf-8") as f:
            precomputed = json.load(f)

        # --- Retrieval comparison ---
        if precomputed.get("retrieval_evaluation"):
            st.subheader("Retrieval: 10 Experiments")
            rows = []
            for r in precomputed["retrieval_evaluation"]:
                rows.append({
                    "Method": r["method"],
                    "Model": r["model"],
                    "Chunking": r["chunking"],
                    "P@5": r["precision_at_5"],
                    "R@5": r["recall_at_5"],
                    "MRR": r["mrr"],
                })
            df_ret = pd.DataFrame(rows)

            # Highlight best row
            def highlight_best(row):
                is_best = (
                    row["Method"] == "Vector"
                    and row["Model"] == "mpnet"
                    and row["Chunking"] == "section_based"
                )
                return [
                    "background-color: #EAF2D7; font-weight: bold" if is_best else ""
                    for _ in row
                ]

            st.dataframe(
                df_ret.style.apply(highlight_best, axis=1).format(
                    {"P@5": "{:.4f}", "R@5": "{:.4f}", "MRR": "{:.4f}"}
                ),
                use_container_width=True,
            )

            st.info(
                "**Best retrieval:** Vector + mpnet + section_based — "
                "MRR 1.0 (perfect first-hit), Recall@5 0.983"
            )

        # --- Generation comparison ---
        if precomputed.get("generation_evaluation"):
            st.subheader("Generation: 3 Prompt Strategies")
            rows = []
            for r in precomputed["generation_evaluation"]:
                rows.append({
                    "Strategy": r["strategy"],
                    "ROUGE-L": r["rouge_l"],
                    "BERTScore F1": r["bert_score_f1"],
                    "Faithfulness": r["faithfulness"],
                    "Avg Time/q (s)": r["avg_time_per_query"],
                })
            df_gen = pd.DataFrame(rows)

            def highlight_structured(row):
                is_best = row["Strategy"] == "structured"
                return [
                    "background-color: #EAF2D7; font-weight: bold" if is_best else ""
                    for _ in row
                ]

            st.dataframe(
                df_gen.style.apply(highlight_structured, axis=1).format({
                    "ROUGE-L": "{:.4f}",
                    "BERTScore F1": "{:.4f}",
                    "Faithfulness": "{:.4f}",
                    "Avg Time/q (s)": "{:.1f}",
                }),
                use_container_width=True,
            )

            st.info(
                "**Best generation:** Structured prompting — "
                "highest faithfulness (0.64), highest ROUGE-L (0.25), fastest (17.4s/q)"
            )

        # --- Key findings ---
        st.subheader("Key Findings")
        st.markdown("""
- **Section-based chunking** (avg 380 words) dominates — enough context for retrieval and generation
- **Paragraph chunking** (avg 75 words) underperforms — chunks too small to carry sufficient context
- **Hybrid RRF did not help** — BM25 noise dragged down the fusion vs pure vector
- **BGE-M3** (1024d, 8192 tokens) matched mpnet but at 6x compute cost — not justified
- **Few-shot prompting** underperformed — examples consumed context and confused the small 0.5B model
- **BM25** was weakest — keyword matching fails on semantic queries like "history and origin of baklava"
        """)

        # --- Winner box ---
        st.success(
            "**Winner: Vector + mpnet + section_based + structured prompting**\n\n"
            "P@5=0.653 | R@5=0.983 | MRR=1.000 | ROUGE-L=0.246 | Faithfulness=0.640"
        )
    else:
        st.warning(
            "No pre-computed results found. Run `python evaluator.py` first."
        )

# ---------------------------------------------------------------
# TAB 5: About
# ---------------------------------------------------------------
with tab_about:
    st.header("About This System")

    st.markdown("""
    ### Mediterranean Cuisine RAG System

    A Retrieval-Augmented Generation (RAG) pipeline built for **COMP 647-02:
    Transforming Text into Meaning**.

    #### Pipeline Components

    | Component | Description |
    |-----------|-------------|
    | **Chunking** | 5 strategies: section-based, fixed-200, fixed-500, sentence-based, paragraph |
    | **Embedding** | 4 models: MiniLM, MPNet, BGE-small, BGE-M3 via SentenceTransformers + FAISS |
    | **Retrieval** | Vector (cosine similarity), BM25 (keyword), Hybrid RRF fusion |
    | **Generation** | Qwen2.5-0.5B-Instruct with zero-shot, few-shot, structured prompts |
    | **Evaluation** | Precision@5, Recall@5, MRR, ROUGE-L, BERTScore, Faithfulness |

    #### Architecture
    - **Corpus**: Wikipedia articles on Mediterranean cuisine (hummus, falafel, couscous, tagine, baklava, tabbouleh, paella, Greek cuisine, etc.)
    - **Vector store**: FAISS with cosine similarity (Inner Product on normalized vectors)
    - **LLM**: Qwen2.5-0.5B-Instruct running on CPU with float32
    - **Fusion**: Reciprocal Rank Fusion (RRF) with k=60

    #### Key Findings
    - **Vector retrieval** achieves near-perfect recall (0.98) and MRR (1.0) on section-based chunks
    - **BM25 struggles** with semantic queries (e.g., "history and origin of baklava")
    - **Hybrid RRF** combines the best of both but doesn't always beat pure vector
    - **Few-shot prompting** causes example contamination in the 0.5B model
    - **Structured prompting** produces the best ROUGE-L scores

    ---
    **Team**: COMP 647-02 Group
    """)
