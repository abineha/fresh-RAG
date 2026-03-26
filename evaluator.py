"""
Mediterranean Cuisine -- RAG Evaluation Pipeline
==================================================
Deliverable 5: Evaluation

Evaluates both retrieval quality and generation quality:
  A) Retrieval metrics:  Precision@5, Recall@5, MRR
  B) Generation metrics: ROUGE-L, BERTScore F1, Faithfulness

Usage:
    python evaluator.py                    # run full evaluation
    python evaluator.py --retrieval-only   # only retrieval metrics
    python evaluator.py --generation-only  # only generation metrics
"""

import os
import json
import argparse
import re
from collections import defaultdict

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
GOLD_FILE = "rag_benchmark_answers.json"

RETRIEVAL_DIR = "retrieval_results"
GENERATION_DIR = "generation_results"
RESULTS_DIR = "evaluation_results"

CHUNK_FILES = {
    "section_based":  "chunks.json",
    "fixed_200":      "chunks_fixed_200.json",
    "fixed_500":      "chunks_fixed_500.json",
    "sentence_based": "chunks_sentence.json",
}

# Retrieval experiments to evaluate
RETRIEVAL_EXPERIMENTS = [
    ("retrieval_vector_mpnet_section_based.json",   "Vector",     "mpnet", "section_based"),
    ("retrieval_bm25_none_section_based.json",      "BM25",       "none",  "section_based"),
    ("retrieval_hybrid_mpnet_section_based.json",    "Hybrid RRF", "mpnet", "section_based"),
    ("retrieval_vector_mpnet_fixed_200.json",        "Vector",     "mpnet", "fixed_200"),
    ("retrieval_hybrid_mpnet_fixed_200.json",        "Hybrid RRF", "mpnet", "fixed_200"),
    ("retrieval_vector_bge_section_based.json",      "Vector",     "bge",   "section_based"),
]

# Generation experiments to evaluate
GENERATION_EXPERIMENTS = [
    ("generation_results_zero_shot.json",   "zero_shot"),
    ("generation_results_few_shot.json",    "few_shot"),
    ("generation_results_structured.json",  "structured"),
]

# Chunk matching threshold (fraction of gold words that must appear in retrieved chunk)
MATCH_THRESHOLD = 0.5

# Stopwords for faithfulness check
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "but", "and", "or", "if", "while", "because", "until", "that", "which",
    "who", "whom", "this", "these", "those", "it", "its", "his", "her",
    "their", "my", "your", "our", "he", "she", "they", "we", "you", "i",
    "me", "him", "us", "them", "what", "also", "about", "up", "any",
}


# ---------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------

def load_gold_standard():
    """Load gold-standard benchmark answers."""
    with open(GOLD_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def load_retrieval_results(filename):
    """Load retrieval results from JSON file."""
    filepath = os.path.join(RETRIEVAL_DIR, filename)
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_generation_results(filename):
    """Load generation results from JSON file."""
    filepath = os.path.join(GENERATION_DIR, filename)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def load_chunks(strategy):
    """Load chunk texts for a given chunking strategy."""
    chunk_file = CHUNK_FILES[strategy]
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


# ---------------------------------------------------------------
# STEP 2: CHUNK MATCHING (fuzzy word overlap)
# ---------------------------------------------------------------

def get_content_words(text):
    """Extract lowercase content words from text (no stopwords)."""
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def chunks_match(gold_text, retrieved_text, threshold=MATCH_THRESHOLD):
    """Check if a gold-standard chunk matches a retrieved chunk by word overlap.

    Returns True if at least `threshold` fraction of gold content words
    appear in the retrieved chunk text.
    """
    gold_words = set(get_content_words(gold_text))
    if not gold_words:
        return False
    retrieved_words = set(get_content_words(retrieved_text))
    overlap = len(gold_words & retrieved_words)
    return (overlap / len(gold_words)) >= threshold


# ---------------------------------------------------------------
# STEP 3: RETRIEVAL METRICS
# ---------------------------------------------------------------

def evaluate_retrieval(retrieval_results, gold_standard, chunks_lookup):
    """Compute Precision@5, Recall@5, and MRR for one retrieval experiment.

    Returns per-query metrics and averages.
    """
    per_query = []

    for gold in gold_standard:
        qid = gold["query_id"]
        gold_chunks = gold["retrieved_context"]

        # Find matching retrieval result
        retr = None
        for r in retrieval_results:
            if str(r["query_id"]) == str(qid):
                retr = r
                break

        if retr is None:
            per_query.append({
                "query_id": qid,
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "reciprocal_rank": 0.0,
            })
            continue

        # Get retrieved chunk texts
        retrieved_texts = []
        for hit in retr["retrieved"]:
            cid = hit["chunk_id"]
            chunk = chunks_lookup.get(cid, {})
            retrieved_texts.append(chunk.get("text", ""))

        # Count matches
        matched_gold = 0
        first_match_rank = None

        for rank, ret_text in enumerate(retrieved_texts, 1):
            is_match = False
            for gc in gold_chunks:
                if chunks_match(gc["text"], ret_text):
                    is_match = True
                    break

            if is_match and first_match_rank is None:
                first_match_rank = rank

        # For precision/recall, count unique gold chunks matched
        matched_gold = 0
        for gc in gold_chunks:
            for ret_text in retrieved_texts:
                if chunks_match(gc["text"], ret_text):
                    matched_gold += 1
                    break

        # Also count how many retrieved chunks match any gold chunk
        matched_retrieved = 0
        for ret_text in retrieved_texts:
            for gc in gold_chunks:
                if chunks_match(gc["text"], ret_text):
                    matched_retrieved += 1
                    break

        precision = matched_retrieved / len(retrieved_texts) if retrieved_texts else 0.0
        recall = matched_gold / len(gold_chunks) if gold_chunks else 0.0
        rr = 1.0 / first_match_rank if first_match_rank else 0.0

        per_query.append({
            "query_id": qid,
            "precision_at_5": round(precision, 4),
            "recall_at_5": round(recall, 4),
            "reciprocal_rank": round(rr, 4),
        })

    # Compute averages
    n = len(per_query)
    avg_precision = sum(q["precision_at_5"] for q in per_query) / n
    avg_recall = sum(q["recall_at_5"] for q in per_query) / n
    avg_mrr = sum(q["reciprocal_rank"] for q in per_query) / n

    return {
        "per_query": per_query,
        "avg_precision_at_5": round(avg_precision, 4),
        "avg_recall_at_5": round(avg_recall, 4),
        "mrr": round(avg_mrr, 4),
    }


# ---------------------------------------------------------------
# STEP 4: GENERATION METRICS — ROUGE-L
# ---------------------------------------------------------------

def compute_rouge_l(gold_answers, generated_answers):
    """Compute ROUGE-L F1 for each query pair. Returns per-query and average."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for gold, gen in zip(gold_answers, generated_answers):
        result = scorer.score(gold, gen)
        scores.append(round(result['rougeL'].fmeasure, 4))

    avg = round(sum(scores) / len(scores), 4) if scores else 0.0
    return scores, avg


# ---------------------------------------------------------------
# STEP 5: GENERATION METRICS — BERTScore
# ---------------------------------------------------------------

def compute_bert_score(gold_answers, generated_answers):
    """Compute BERTScore F1 for each query pair. Returns per-query and average."""
    from bert_score import score

    P, R, F1 = score(
        generated_answers,
        gold_answers,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )

    scores = [round(f.item(), 4) for f in F1]
    avg = round(sum(scores) / len(scores), 4) if scores else 0.0
    return scores, avg


# ---------------------------------------------------------------
# STEP 6: FAITHFULNESS CHECK
# ---------------------------------------------------------------

def compute_faithfulness(generated_answers, retrieved_contexts):
    """Check what fraction of content words in the answer appear in the context.

    For each answer, compute:
        grounding_ratio = |answer_words ∩ context_words| / |answer_words|

    Returns per-query ratios and average.
    """
    ratios = []

    for answer, contexts in zip(generated_answers, retrieved_contexts):
        answer_words = set(get_content_words(answer))
        if not answer_words:
            ratios.append(1.0)
            continue

        # Combine all retrieved chunk texts
        context_text = " ".join(c.get("text", "") for c in contexts)
        context_words = set(get_content_words(context_text))

        grounded = len(answer_words & context_words)
        ratio = grounded / len(answer_words)
        ratios.append(round(ratio, 4))

    avg = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    return ratios, avg


# ---------------------------------------------------------------
# STEP 7: RUN FULL EVALUATION
# ---------------------------------------------------------------

def run_retrieval_evaluation(gold_standard):
    """Evaluate all 6 retrieval experiments."""
    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION")
    print("=" * 70)

    all_results = []

    for filename, method, model, strategy in RETRIEVAL_EXPERIMENTS:
        filepath = os.path.join(RETRIEVAL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP: {filename} not found")
            continue

        retrieval_results = load_retrieval_results(filename)
        chunks_lookup = load_chunks(strategy)

        metrics = evaluate_retrieval(retrieval_results, gold_standard, chunks_lookup)

        result = {
            "file": filename,
            "method": method,
            "model": model,
            "chunking": strategy,
            "precision_at_5": metrics["avg_precision_at_5"],
            "recall_at_5": metrics["avg_recall_at_5"],
            "mrr": metrics["mrr"],
            "per_query": metrics["per_query"],
        }
        all_results.append(result)

        print(f"\n  {method:12s} | {model:5s} | {strategy:14s}")
        print(f"    Precision@5: {metrics['avg_precision_at_5']:.4f}")
        print(f"    Recall@5:    {metrics['avg_recall_at_5']:.4f}")
        print(f"    MRR:         {metrics['mrr']:.4f}")

    # Print comparison table
    print(f"\n  {'-' * 70}")
    print(f"  {'Method':<12s} {'Model':<6s} {'Chunking':<15s} {'P@5':>6s} {'R@5':>6s} {'MRR':>6s}")
    print(f"  {'-' * 70}")
    for r in all_results:
        print(f"  {r['method']:<12s} {r['model']:<6s} {r['chunking']:<15s} "
              f"{r['precision_at_5']:>6.4f} {r['recall_at_5']:>6.4f} {r['mrr']:>6.4f}")
    print(f"  {'-' * 70}")

    return all_results


def run_generation_evaluation(gold_standard):
    """Evaluate all 3 generation experiments."""
    print("\n" + "=" * 70)
    print("  GENERATION EVALUATION")
    print("=" * 70)

    # Prepare gold answers (same for all strategies)
    gold_answers = [g["response"] for g in gold_standard]

    all_results = []

    for filename, strategy in GENERATION_EXPERIMENTS:
        filepath = os.path.join(GENERATION_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP: {filename} not found")
            continue

        print(f"\n  Evaluating: {strategy}")
        gen_results = load_generation_results(filename)

        # Sort by query_id to match gold order
        gen_results.sort(key=lambda x: int(x["query_id"]))
        generated_answers = [g["response"] for g in gen_results]
        retrieved_contexts = [g["retrieved_context"] for g in gen_results]

        # ROUGE-L
        print("    Computing ROUGE-L...")
        rouge_scores, rouge_avg = compute_rouge_l(gold_answers, generated_answers)

        # BERTScore
        print("    Computing BERTScore...")
        bert_scores, bert_avg = compute_bert_score(gold_answers, generated_answers)

        # Faithfulness
        print("    Computing Faithfulness...")
        faith_scores, faith_avg = compute_faithfulness(generated_answers, retrieved_contexts)

        # Get average generation time from results
        avg_time = sum(g.get("generation_time_s", 0) for g in gen_results) / len(gen_results)

        result = {
            "strategy": strategy,
            "rouge_l": rouge_avg,
            "bert_score_f1": bert_avg,
            "faithfulness": faith_avg,
            "avg_time_per_query": round(avg_time, 1),
            "per_query": [
                {
                    "query_id": gen_results[i]["query_id"],
                    "rouge_l": rouge_scores[i],
                    "bert_score_f1": bert_scores[i],
                    "faithfulness": faith_scores[i],
                }
                for i in range(len(gen_results))
            ],
        }
        all_results.append(result)

        print(f"    ROUGE-L:       {rouge_avg:.4f}")
        print(f"    BERTScore F1:  {bert_avg:.4f}")
        print(f"    Faithfulness:  {faith_avg:.4f}")
        print(f"    Avg time/q:    {avg_time:.1f}s")

    # Print comparison table
    print(f"\n  {'-' * 70}")
    print(f"  {'Strategy':<14s} {'ROUGE-L':>8s} {'BERTScore':>10s} {'Faithful':>10s} {'Time/q':>8s}")
    print(f"  {'-' * 70}")
    for r in all_results:
        print(f"  {r['strategy']:<14s} {r['rouge_l']:>8.4f} {r['bert_score_f1']:>10.4f} "
              f"{r['faithfulness']:>10.4f} {r['avg_time_per_query']:>7.1f}s")
    print(f"  {'-' * 70}")

    return all_results


def save_results(retrieval_results, generation_results):
    """Save all evaluation results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "retrieval_evaluation": retrieval_results,
        "generation_evaluation": generation_results,
    }

    filepath = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {filepath}")

    return filepath


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline: retrieval + generation quality"
    )
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only run retrieval evaluation")
    parser.add_argument("--generation-only", action="store_true",
                        help="Only run generation evaluation")
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD,
                        help=f"Word overlap threshold for chunk matching (default: {MATCH_THRESHOLD})")
    args = parser.parse_args()

    # Update threshold if specified
    MATCH_THRESHOLD = args.match_threshold

    # Load gold standard
    gold_standard = load_gold_standard()
    print(f"  Loaded {len(gold_standard)} gold-standard queries from {GOLD_FILE}")

    retrieval_results = []
    generation_results = []

    if not args.generation_only:
        retrieval_results = run_retrieval_evaluation(gold_standard)

    if not args.retrieval_only:
        generation_results = run_generation_evaluation(gold_standard)

    # Save
    save_results(retrieval_results, generation_results)

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'=' * 70}")
