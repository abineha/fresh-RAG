"""
Mediterranean Cuisine — RAG Chunking Pipeline
==============================================
Deliverable 2: Chunking

Reads individual corpus files from corpus/, cleans them,
splits into retrieval-ready chunks, and writes chunks.json.

Supports four strategies:
  A) section_based  — respects document structure (primary)
  B) fixed_size     — fixed N-word chunks with overlap
  C) sentence_based — groups sentences to ~300 words
  D) paragraph      — splits at paragraph boundaries with sentence overlap

Usage:
    python chunker.py                           # default: section_based
    python chunker.py --strategy fixed_size --chunk-size 300
    python chunker.py --strategy sentence_based
    python chunker.py --strategy paragraph
    python chunker.py --strategy all            # run all strategies
"""

import os
import re
import json
import hashlib
import argparse
from glob import glob

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
CORPUS_DIR      = "corpus"
OUTPUT_DIR      = "."
MIN_CHUNK_WORDS = 100
MAX_CHUNK_WORDS = 500
OVERLAP_WORDS   = 50
SENTENCE_TARGET = 300


def slugify(s: str) -> str:
    """Convert a title string to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9_]", "_", s.lower())[:60]


# --------------------------------------------------------------
# STEP 1: PARSE METADATA HEADER
# --------------------------------------------------------------

def parse_file(filepath: str) -> dict | None:
    """Parse a corpus file into metadata dict + body text.

    Returns {"title": ..., "source": ..., "url": ..., "body": ...}
    or None if the file can't be parsed.
    """
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()

    # Split on the ==== separator line
    lines = raw.split("\n")
    sep_idx = None
    for i, line in enumerate(lines):
        if line.startswith("===="):
            sep_idx = i
            break

    if sep_idx is None:
        print(f"  [WARN] No separator found in {filepath}")
        return None

    # Parse header fields
    header_lines = lines[:sep_idx]
    meta = {}
    for hl in header_lines:
        if ":" in hl:
            key, val = hl.split(":", 1)
            meta[key.strip().lower()] = val.strip()

    # Body is everything after the separator (skip blank line after it)
    body_start = sep_idx + 1
    if body_start < len(lines) and lines[body_start].strip() == "":
        body_start += 1
    body = "\n".join(lines[body_start:])

    return {
        "title":  meta.get("title", os.path.basename(filepath)),
        "source": meta.get("source", "unknown"),
        "url":    meta.get("url", ""),
        "body":   body,
    }


# --------------------------------------------------------------
# STEP 2: CLEAN BODY TEXT
# --------------------------------------------------------------

# Wikibooks navigation preamble (appears at the top of every Wikibooks file)
_WIKIBOOKS_PREAMBLE = re.compile(
    r"^Cookbook\s*\|.*(?:Recipes|Ingredients|Equipment|Techniques).*$",
    re.MULTILINE,
)

# Blog "List of cuisines" numbered block (01. ... through 79. ...)
_BLOG_CUISINE_LIST = re.compile(
    r"(?:^List of cuisines\s*$\n)"        # optional heading
    r"(?:^\s*$\n)*"                        # blank lines
    r"(?:^\d{2}\.\s+.+$\n(?:^\s*$\n)*)+", # numbered entries with blanks
    re.MULTILINE,
)

# Blog metadata lines (date, author, comments)
_BLOG_META = re.compile(
    r"(?:^\w+ \d{1,2}, \d{4}\s*$|"    # dates like "October 5, 2017"
    r"^\s*/\s*$|"                       # bare slashes
    r"^laurarose1990\s*$|"              # author name
    r"^\d+ Comments?\s*$)",             # comment count
    re.MULTILINE,
)

# Wikibooks reference lines starting with ↑
_WIKIBOOKS_REFS = re.compile(r"^↑\s+.+$", re.MULTILINE)


def clean_body(text: str, source: str) -> str:
    """Remove noise from body text based on source type."""

    if source == "wikibooks":
        # Remove navigation preamble
        text = _WIKIBOOKS_PREAMBLE.sub("", text)
        # Remove reference lines
        text = _WIKIBOOKS_REFS.sub("", text)
        # Remove "References" heading if it's now orphaned
        text = re.sub(r"(?m)^References\s*$", "", text)

    elif source == "blog":
        # Remove blog metadata lines
        text = _BLOG_META.sub("", text)
        # Remove the "List of cuisines" numbered block
        text = _BLOG_CUISINE_LIST.sub("", text)
        # Deduplicate paragraphs (content appears 2-3x in blog files)
        text = _deduplicate_paragraphs(text)

    # Universal cleanup: collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def _deduplicate_paragraphs(text: str) -> str:
    """Remove duplicate paragraphs using content hashing."""
    paragraphs = re.split(r"\n\n+", text)
    seen = set()
    unique = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        # Hash the paragraph (normalise whitespace for comparison)
        normalised = re.sub(r"\s+", " ", stripped)
        h = hashlib.md5(normalised.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(stripped)
    return "\n\n".join(unique)


# --------------------------------------------------------------
# STEP 3: DETECT SECTION BOUNDARIES
# --------------------------------------------------------------

def _is_heading(line: str) -> bool:
    """Heuristic: a line is a heading if it's short, has no trailing
    punctuation, and is not just whitespace or a number."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    # Must not end with sentence/list punctuation
    if stripped[-1] in ".;,!?:":
        return False
    # Must not be a pure number like "33"
    if stripped.replace(".", "").strip().isdigit():
        return False
    # Must have at least one letter
    if not any(c.isalpha() for c in stripped):
        return False
    # Headings are short — typically 1-6 words
    if len(stripped.split()) > 7:
        return False
    return True


def detect_sections(text: str, source: str) -> list[dict]:
    """Split cleaned text into sections: [{"heading": ..., "body": ...}, ...]

    Strategy varies by source type:
      - wikibooks/blog: split on detected headings
      - wikipedia: group paragraphs by word count
    """
    if source in ("wikibooks", "blog"):
        return _split_on_headings(text)
    else:
        return _group_paragraphs(text)


def _split_on_headings(text: str) -> list[dict]:
    """Split text at heading lines (wikibooks & blog)."""
    paragraphs = re.split(r"\n\n+", text)
    sections = []
    current_heading = "Introduction"
    current_body_parts = []

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if _is_heading(stripped):
            # Flush previous section
            if current_body_parts:
                sections.append({
                    "heading": current_heading,
                    "body": "\n\n".join(current_body_parts),
                })
            # Clean heading: strip leading numbers like "33. "
            current_heading = re.sub(r"^\d+\.\s*", "", stripped) or stripped
            current_body_parts = []
        else:
            current_body_parts.append(stripped)

    # Flush last section
    if current_body_parts:
        sections.append({
            "heading": current_heading,
            "body": "\n\n".join(current_body_parts),
        })

    return sections


def _group_paragraphs(text: str, target_words: int = MAX_CHUNK_WORDS) -> list[dict]:
    """Group consecutive paragraphs until hitting target word count (wikipedia)."""
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    sections = []
    current_parts = []
    current_words = 0

    for para in paragraphs:
        pw = len(para.split())
        # If adding this paragraph would exceed target AND we already have content,
        # start a new section
        if current_words + pw > target_words and current_parts:
            sections.append({
                "heading": "General",
                "body": "\n\n".join(current_parts),
            })
            current_parts = []
            current_words = 0
        current_parts.append(para)
        current_words += pw

    if current_parts:
        sections.append({
            "heading": "General",
            "body": "\n\n".join(current_parts),
        })

    return sections


# --------------------------------------------------------------
# STEPS 4–5: ADAPTIVE SIZING + OVERLAP
# --------------------------------------------------------------

def _split_text_with_overlap(text: str, max_words: int, overlap: int,
                             min_tail: int = MIN_CHUNK_WORDS) -> list[str]:
    """Split text into sub-chunks of ~max_words with overlap words repeated.
    If the final tail chunk would be smaller than min_tail, merge it into
    the previous chunk instead of creating a tiny orphan."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        # If the remaining tail after this chunk would be too small, absorb it
        remaining = len(words) - end
        if 0 < remaining < min_tail:
            end = len(words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def apply_adaptive_sizing(sections: list[dict],
                          min_words: int = MIN_CHUNK_WORDS,
                          max_words: int = MAX_CHUNK_WORDS,
                          overlap: int = OVERLAP_WORDS) -> list[dict]:
    """Apply adaptive sizing rules to sections.

    - Small sections (<min_words): merge with adjacent section
    - Good sections (min_words–max_words): keep as-is
    - Large sections (>max_words): split with overlap

    Returns list of {"heading": ..., "body": ...} ready for chunking.
    """
    if not sections:
        return []

    # Phase 1: merge small sections forward (or backward if last)
    merged = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        wc = len(sec["body"].split())
        if wc < min_words and i + 1 < len(sections):
            # Merge forward: append this body to the next section
            next_sec = sections[i + 1]
            sections[i + 1] = {
                "heading": next_sec["heading"],
                "body": sec["body"] + "\n\n" + next_sec["body"],
            }
        elif wc < min_words and merged:
            # Merge backward: append to previous
            prev = merged[-1]
            merged[-1] = {
                "heading": prev["heading"],
                "body": prev["body"] + "\n\n" + sec["body"],
            }
        else:
            merged.append(sec)
        i += 1

    # Phase 2: split large sections
    result = []
    for sec in merged:
        wc = len(sec["body"].split())
        if wc > max_words:
            sub_chunks = _split_text_with_overlap(sec["body"], max_words, overlap)
            for j, chunk_text in enumerate(sub_chunks):
                suffix = f" (part {j+1})" if len(sub_chunks) > 1 else ""
                result.append({
                    "heading": sec["heading"] + suffix,
                    "body": chunk_text,
                })
        else:
            result.append(sec)

    return result


# --------------------------------------------------------------
# STEP 6: CONTEXT HEADER
# --------------------------------------------------------------

def make_chunk_text(body: str, source: str, title: str, section: str) -> str:
    """Prepend the context header to chunk body text."""
    header = f"[Source: {source} | Title: {title} | Section: {section}]"
    return f"{header}\n{body}"


# --------------------------------------------------------------
# STEP 7: FULL SECTION-BASED PIPELINE
# --------------------------------------------------------------

def chunk_section_based(docs: list[dict]) -> list[dict]:
    """Strategy A: section-based chunking with adaptive sizing."""
    all_chunks = []

    for doc in docs:
        title  = doc["title"]
        source = doc["source"]
        url    = doc["url"]
        body   = doc["body"]

        cleaned = clean_body(body, source)
        if not cleaned or len(cleaned.split()) < 20:
            continue

        sections = detect_sections(cleaned, source)
        sized    = apply_adaptive_sizing(sections)

        slug = slugify(title)
        for i, sec in enumerate(sized):
            text = make_chunk_text(sec["body"], source, title, sec["heading"])
            all_chunks.append({
                "chunk_id":       f"{source}_{slug}_{i:03d}",
                "doc_title":      title,
                "source":         source,
                "url":            url,
                "section":        sec["heading"],
                "text":           text,
                "word_count":     len(sec["body"].split()),
                "chunk_strategy": "section_based",
            })

    return all_chunks


# --------------------------------------------------------------
# STRATEGY B: FIXED-SIZE CHUNKING
# --------------------------------------------------------------

def chunk_fixed_size(docs: list[dict], chunk_size: int = 300,
                     overlap: int = OVERLAP_WORDS) -> list[dict]:
    """Strategy B: fixed N-word chunks with overlap, ignoring structure."""
    all_chunks = []

    for doc in docs:
        title  = doc["title"]
        source = doc["source"]
        url    = doc["url"]
        body   = doc["body"]

        cleaned = clean_body(body, source)
        if not cleaned or len(cleaned.split()) < 20:
            continue

        sub_chunks = _split_text_with_overlap(cleaned, chunk_size, overlap)
        slug = slugify(title)
        for i, chunk_text in enumerate(sub_chunks):
            text = make_chunk_text(chunk_text, source, title, "General")
            all_chunks.append({
                "chunk_id":       f"{source}_{slug}_{i:03d}",
                "doc_title":      title,
                "source":         source,
                "url":            url,
                "section":        "General",
                "text":           text,
                "word_count":     len(chunk_text.split()),
                "chunk_strategy": f"fixed_size_{chunk_size}",
            })

    return all_chunks


# --------------------------------------------------------------
# STRATEGY C: SENTENCE-BASED CHUNKING
# --------------------------------------------------------------

def chunk_sentence_based(docs: list[dict],
                         target_words: int = SENTENCE_TARGET) -> list[dict]:
    """Strategy C: group sentences to ~target_words, no overlap."""
    all_chunks = []

    for doc in docs:
        title  = doc["title"]
        source = doc["source"]
        url    = doc["url"]
        body   = doc["body"]

        cleaned = clean_body(body, source)
        if not cleaned or len(cleaned.split()) < 20:
            continue

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        slug = slugify(title)

        current_sentences = []
        current_words = 0
        chunk_idx = 0

        for sent in sentences:
            sw = len(sent.split())
            if current_words + sw > target_words and current_sentences:
                chunk_body = " ".join(current_sentences)
                text = make_chunk_text(chunk_body, source, title, "General")
                all_chunks.append({
                    "chunk_id":       f"{source}_{slug}_{chunk_idx:03d}",
                    "doc_title":      title,
                    "source":         source,
                    "url":            url,
                    "section":        "General",
                    "text":           text,
                    "word_count":     len(chunk_body.split()),
                    "chunk_strategy": "sentence_based",
                })
                chunk_idx += 1
                current_sentences = []
                current_words = 0
            current_sentences.append(sent)
            current_words += sw

        # Flush remaining
        if current_sentences:
            chunk_body = " ".join(current_sentences)
            # Merge tiny trailing chunk with previous if possible
            if len(chunk_body.split()) < MIN_CHUNK_WORDS and all_chunks and \
               all_chunks[-1]["doc_title"] == title:
                prev = all_chunks[-1]
                # Strip header from prev text, merge, re-add header
                prev_body = prev["text"].split("\n", 1)[1]
                merged_body = prev_body + " " + chunk_body
                prev["text"] = make_chunk_text(merged_body, source, title, "General")
                prev["word_count"] = len(merged_body.split())
            else:
                text = make_chunk_text(chunk_body, source, title, "General")
                all_chunks.append({
                    "chunk_id":       f"{source}_{slug}_{chunk_idx:03d}",
                    "doc_title":      title,
                    "source":         source,
                    "url":            url,
                    "section":        "General",
                    "text":           text,
                    "word_count":     len(chunk_body.split()),
                    "chunk_strategy": "sentence_based",
                })

    return all_chunks


# --------------------------------------------------------------
# STRATEGY D: PARAGRAPH-BASED CHUNKING
# --------------------------------------------------------------
# Split documents at paragraph boundaries (\n\n), preserve headings,
# sentence overlap, then enforce a max character limit.
#
# Why paragraph-based:
# - Wikipedia articles have clear paragraph structure with logical topics
# - Blog posts are written in natural narrative paragraphs
# - Wikibooks recipes have natural sections (ingredients, procedure, notes)
# - Splitting at paragraphs preserves semantic coherence better than fixed size
#
# Max size limit prevents very long paragraphs from becoming chunks too large
# for the embedding model or LLM context window.
# Too short → dropped (most likely noise)
# Too long  → split into multiple chunks with sentence overlap (nothing lost)
# Just right → kept as one chunk

PARA_MAX_CHARS       = 1000
PARA_MIN_CHARS       = 100
PARA_MIN_WORDS       = 25    # also enforce a minimum word count to filter noise
PARA_OVERLAP_SENTS   = 1     # sentences to carry over between split chunks


def _para_ok(text: str, min_chars: int, min_words: int) -> bool:
    """Check if a paragraph/chunk meets both minimum character and word thresholds."""
    return len(text) >= min_chars and len(text.split()) >= min_words


def chunk_paragraph(docs: list[dict],
                    max_chars: int = PARA_MAX_CHARS,
                    min_chars: int = PARA_MIN_CHARS,
                    min_words: int = PARA_MIN_WORDS,
                    overlap: int = PARA_OVERLAP_SENTS) -> list[dict]:
    """Strategy D: paragraph-based chunking with heading preservation
    and sentence overlap for long-paragraph splits."""
    all_chunks = []

    for doc in docs:
        title  = doc["title"]
        source = doc["source"]
        url    = doc["url"]
        body   = doc["body"]

        cleaned = clean_body(body, source)
        if not cleaned or len(cleaned.split()) < 20:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
        slug = slugify(title)
        chunk_idx = 0

        for para in paragraphs:
            if len(para) <= max_chars:
                # Paragraph fits — keep if long enough (both chars and words)
                if _para_ok(para, min_chars, min_words):
                    text = make_chunk_text(para, source, title, "General")
                    all_chunks.append({
                        "chunk_id":       f"{source}_{slug}_{chunk_idx:03d}",
                        "doc_title":      title,
                        "source":         source,
                        "url":            url,
                        "section":        "General",
                        "text":           text,
                        "word_count":     len(para.split()),
                        "chunk_strategy": "paragraph",
                    })
                    chunk_idx += 1
            else:
                # Paragraph too long — split at sentence boundaries with overlap
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                overlap_buffer = []

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_chars:
                        current_chunk = (current_chunk + " " + sentence).strip() \
                                        if current_chunk else sentence
                    else:
                        if _para_ok(current_chunk, min_chars, min_words):
                            text = make_chunk_text(current_chunk, source, title, "General")
                            all_chunks.append({
                                "chunk_id":       f"{source}_{slug}_{chunk_idx:03d}",
                                "doc_title":      title,
                                "source":         source,
                                "url":            url,
                                "section":        "General",
                                "text":           text,
                                "word_count":     len(current_chunk.split()),
                                "chunk_strategy": "paragraph",
                            })
                            chunk_idx += 1
                            # Carry last N sentences into next chunk
                            overlap_buffer = re.split(
                                r"(?<=[.!?])\s+", current_chunk
                            )[-overlap:]
                        current_chunk = " ".join(overlap_buffer + [sentence])

                # Flush remaining
                if _para_ok(current_chunk, min_chars, min_words):
                    text = make_chunk_text(current_chunk, source, title, "General")
                    all_chunks.append({
                        "chunk_id":       f"{source}_{slug}_{chunk_idx:03d}",
                        "doc_title":      title,
                        "source":         source,
                        "url":            url,
                        "section":        "General",
                        "text":           text,
                        "word_count":     len(current_chunk.split()),
                        "chunk_strategy": "paragraph",
                    })
                    chunk_idx += 1

    return all_chunks


# --------------------------------------------------------------
# SUMMARY + OUTPUT
# --------------------------------------------------------------

def print_summary(chunks: list[dict], strategy: str):
    """Print a summary of the chunking results."""
    if not chunks:
        print("  No chunks produced!")
        return
    word_counts = [c["word_count"] for c in chunks]
    sources = {}
    for c in chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1

    print(f"\n{'-'*50}")
    print(f"  Strategy           : {strategy}")
    print(f"  Total chunks       : {len(chunks)}")
    print(f"  Avg words/chunk    : {sum(word_counts)/len(word_counts):.0f}")
    print(f"  Min words/chunk    : {min(word_counts)}")
    print(f"  Max words/chunk    : {max(word_counts)}")
    print(f"  Chunks by source   :")
    for src, count in sorted(sources.items()):
        print(f"    {src:12s}: {count}")
    print(f"{'-'*50}")


def write_chunks(chunks: list[dict], filepath: str):
    """Write chunks to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(chunks)} chunks to {filepath}")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk the Mediterranean cuisine corpus")
    parser.add_argument("--strategy", default="section_based",
                        choices=["section_based", "fixed_size", "sentence_based", "paragraph", "all"],
                        help="Chunking strategy to use")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Chunk size in words (for fixed_size strategy)")
    args = parser.parse_args()

    # Load all documents
    files = sorted(glob(os.path.join(CORPUS_DIR, "*.txt")))
    print(f"Found {len(files)} corpus files")

    docs = []
    for fp in files:
        result = parse_file(fp)
        if result:
            docs.append(result)
    print(f"Parsed {len(docs)} documents\n")

    if args.strategy == "all":
        # Run all strategies
        for strat, func, outfile in [
            ("section_based",  lambda: chunk_section_based(docs),            "chunks.json"),
            ("fixed_size_200", lambda: chunk_fixed_size(docs, 200),          "chunks_fixed_200.json"),
            ("fixed_size_500", lambda: chunk_fixed_size(docs, 500),          "chunks_fixed_500.json"),
            ("sentence_based", lambda: chunk_sentence_based(docs),           "chunks_sentence.json"),
            ("paragraph",      lambda: chunk_paragraph(docs),                "chunks_paragraph.json"),
        ]:
            chunks = func()
            print_summary(chunks, strat)
            write_chunks(chunks, os.path.join(OUTPUT_DIR, outfile))
            print()
    else:
        if args.strategy == "section_based":
            chunks = chunk_section_based(docs)
            outfile = "chunks.json"
        elif args.strategy == "fixed_size":
            chunks = chunk_fixed_size(docs, args.chunk_size)
            outfile = f"chunks_fixed_{args.chunk_size}.json"
        elif args.strategy == "sentence_based":
            chunks = chunk_sentence_based(docs)
            outfile = "chunks_sentence.json"
        elif args.strategy == "paragraph":
            chunks = chunk_paragraph(docs)
            outfile = "chunks_paragraph.json"

        print_summary(chunks, args.strategy)
        write_chunks(chunks, os.path.join(OUTPUT_DIR, outfile))
