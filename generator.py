"""
Mediterranean Cuisine -- RAG Generation Pipeline
=================================================
Deliverable 4: Prompting and Generation

Uses Qwen/Qwen2.5-0.5B-Instruct to generate answers from retrieved chunks.
Three prompt strategies: zero-shot, few-shot, structured extraction.

Usage:
    python generator.py                                  # default: zero_shot + hybrid retrieval
    python generator.py --prompt-strategy few_shot       # few-shot with example
    python generator.py --prompt-strategy structured     # structured extraction
    python generator.py --run-all                        # run all 3 prompt strategies
    python generator.py --retrieval-file retrieval_results/retrieval_vector_mpnet_section_based.json
"""

import os
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CHUNK_FILES = {
    "section_based":  "chunks.json",
    "fixed_200":      "chunks_fixed_200.json",
    "fixed_500":      "chunks_fixed_500.json",
    "sentence_based": "chunks_sentence.json",
}

DEFAULT_RETRIEVAL_FILE = "retrieval_results/retrieval_hybrid_mpnet_section_based.json"
RESULTS_DIR = "generation_results"

# Generation parameters
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.15,
}

PROMPT_STRATEGIES = ["zero_shot", "few_shot", "structured"]


# ---------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------

SYSTEM_PROMPTS = {
    "zero_shot": (
        "You are a Mediterranean cuisine expert. "
        "Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say \"I don't have enough information to answer this.\" "
        "Keep your answer concise - 2-4 sentences maximum. "
        "Do not make up facts. Do not add information beyond what the context provides."
    ),
    "few_shot": (
        "You are a Mediterranean cuisine expert. "
        "Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say \"I don't have enough information to answer this.\" "
        "Keep your answer concise - 2-4 sentences maximum. "
        "Do not make up facts."
    ),
    "structured": (
        "You are a Mediterranean cuisine expert. "
        "Given the context below, extract the relevant facts that answer the question, then write a short answer. "
        "Rules: "
        "- Use ONLY facts from the context. "
        "- If the answer is not in the context, say \"Not enough information.\" "
        "- Maximum 3 sentences."
    ),
}


# ---------------------------------------------------------------
# STEP 1: LOAD MODEL
# ---------------------------------------------------------------

def load_model():
    """Load Qwen2.5-0.5B-Instruct model and tokenizer for CPU inference."""
    print(f"  Loading model: {MODEL_NAME}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # CPU requires float32
    )
    model.eval()

    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    return model, tokenizer


# ---------------------------------------------------------------
# STEP 2: BUILD PROMPTS
# ---------------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['text']}")
    return "\n\n".join(parts)


def build_user_prompt_zero_shot(context_str: str, question: str) -> str:
    """Zero-shot: context + question, no examples."""
    return f"Context:\n{context_str}\n\nQuestion: {question}"


def build_user_prompt_few_shot(context_str: str, question: str) -> str:
    """Few-shot: one worked example, then the real question."""
    example = (
        "Context:\n"
        "[1] Falafel is a deep-fried ball made from ground chickpeas and herbs, "
        "commonly served in pita bread.\n\n"
        "Question: What is falafel made from?\n\n"
        "Answer: Falafel is made from ground chickpeas and herbs, which are formed "
        "into balls and deep-fried. It is commonly served in pita bread. [1]\n\n"
        "Now answer the following:\n\n"
    )
    return example + f"Context:\n{context_str}\n\nQuestion: {question}"


def build_user_prompt_structured(context_str: str, question: str) -> str:
    """Structured extraction: context + question."""
    return f"Context:\n{context_str}\n\nQuestion: {question}"


USER_PROMPT_BUILDERS = {
    "zero_shot": build_user_prompt_zero_shot,
    "few_shot": build_user_prompt_few_shot,
    "structured": build_user_prompt_structured,
}


def build_messages(strategy: str, context_str: str, question: str) -> list[dict]:
    """Build chat messages for the given prompt strategy."""
    system_prompt = SYSTEM_PROMPTS[strategy]
    user_prompt = USER_PROMPT_BUILDERS[strategy](context_str, question)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------
# STEP 3: GENERATE ANSWER
# ---------------------------------------------------------------

def generate_answer(model, tokenizer, messages: list[dict]) -> tuple[str, float]:
    """Generate an answer from chat messages. Returns (answer, time_seconds)."""
    # Apply Qwen chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    # Decode only the new tokens (skip the prompt)
    new_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return answer, elapsed


# ---------------------------------------------------------------
# STEP 4: RUN GENERATION ON BENCHMARK
# ---------------------------------------------------------------

def run_generation(strategy: str, retrieval_file: str, chunking: str = "section_based"):
    """Run generation for all queries using the specified prompt strategy."""
    # Load retrieval results
    with open(retrieval_file, encoding="utf-8") as f:
        retrieval_results = json.load(f)

    # Load chunks for text lookup
    chunk_file = CHUNK_FILES[chunking]
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)
    chunks_lookup = {c["chunk_id"]: c for c in chunks}

    # Load model
    model, tokenizer = load_model()

    # Extract retrieval metadata from first result
    first = retrieval_results[0]
    retrieval_method = first.get("method", "unknown")
    retrieval_model = first.get("model", "unknown")

    print(f"\n  Strategy: {strategy}")
    print(f"  Retrieval: {retrieval_method} ({retrieval_model})")
    print(f"  Queries: {len(retrieval_results)}")
    print(f"  {'-'*50}")

    all_results = []
    total_time = 0

    for r in retrieval_results:
        query_id = r["query_id"]
        query = r["query"]

        # Get retrieved chunk texts
        retrieved_chunks = []
        for hit in r["retrieved"]:
            cid = hit["chunk_id"]
            chunk = chunks_lookup.get(cid, {})
            retrieved_chunks.append({
                "doc_id": cid,
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "title": chunk.get("doc_title", ""),
            })

        # Build context and messages
        context_str = format_context(retrieved_chunks)
        messages = build_messages(strategy, context_str, query)

        # Generate
        answer, gen_time = generate_answer(model, tokenizer, messages)
        total_time += gen_time

        # Print progress
        preview = answer[:150].encode("ascii", "replace").decode()
        print(f"\n  Q{query_id} ({gen_time:.1f}s): {query}")
        print(f"  A: {preview}...")

        # Save result
        result = {
            "query_id": query_id,
            "query": query,
            "response": answer,
            "retrieved_context": [
                {"doc_id": c["doc_id"], "text": c["text"]}
                for c in retrieved_chunks
            ],
            "prompt_strategy": strategy,
            "retrieval_method": retrieval_method,
            "model": MODEL_NAME,
            "generation_time_s": round(gen_time, 2),
        }
        all_results.append(result)

    print(f"\n  {'-'*50}")
    print(f"  Total generation time: {total_time:.1f}s")
    print(f"  Average per query: {total_time / len(retrieval_results):.1f}s")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {"results": all_results}
    filepath = os.path.join(RESULTS_DIR, f"generation_results_{strategy}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath}")

    return all_results


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers using Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--prompt-strategy", default="zero_shot",
                        choices=PROMPT_STRATEGIES,
                        help="Prompt strategy to use")
    parser.add_argument("--retrieval-file", default=DEFAULT_RETRIEVAL_FILE,
                        help="Path to retrieval results JSON")
    parser.add_argument("--chunking", default="section_based",
                        choices=list(CHUNK_FILES.keys()),
                        help="Chunking strategy (for loading chunk texts)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 3 prompt strategies")
    args = parser.parse_args()

    if args.run_all:
        for strategy in PROMPT_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"  PROMPT STRATEGY: {strategy.upper()}")
            print(f"{'='*60}")
            run_generation(strategy, args.retrieval_file, args.chunking)
        print(f"\n{'='*60}")
        print(f"  ALL STRATEGIES COMPLETE")
        print(f"  Results saved to {RESULTS_DIR}/")
        print(f"{'='*60}")
    else:
        run_generation(args.prompt_strategy, args.retrieval_file, args.chunking)
