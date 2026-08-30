"""
build_testset.py

Creates an evaluation test set by working backwards from chunks we already have.

The problem this solves: to score retrieval, you need questions whose correct
answer you already know. Writing those by hand means reading every paper. So
instead we pick a chunk, ask Claude to write a question that this passage
answers, and record the chunk's id alongside it. The question and its ground
truth are produced together, for free.

Each line of the output is one test case:
    {"question": "...", "expected_chunk_id": "Paper.pdf::chunk7", ...}

Later, run_eval.py asks each question and checks whether expected_chunk_id
came back in the top-k results.

Usage:
    python evals/build_testset.py --n 20

Only calls Anthropic (to write questions). Voyage is not used here, so the
embedding rate limit does not apply -- that only bites when running the eval.
"""

import argparse
import json
import os
import random
import re
import sys

import anthropic
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = "claude-sonnet-4-6"

# A passage full of citation years, "et al.", and arXiv ids is a reference
# list. Questions written from one would test nothing useful -- and worse, no
# retrieval system should be expected to find "which papers are cited" from a
# semantic query. Filtering these out first saves API calls on passages the
# model would reject anyway.
REFERENCEY = re.compile(
    r"\b(19|20)\d{2}[a-z]?\.|et\s+al\.|arXiv:|CoRR,|preprint", re.IGNORECASE
)

MIN_WORDS = 250          # very short chunks rarely contain a full idea
MAX_REFERENCE_RATIO = 12  # citation markers per 100 words before we call it a bibliography

PROMPT = """Below is a passage from a research paper.

Write ONE question that this passage answers. Requirements:
- It must be answerable using only this passage.
- It must be specific enough that a different passage would not also answer it.
  Avoid generic questions like "what is retrieval augmented generation?".
- Ask it the way a researcher would, without referring to "the passage" or
  "this paper" -- someone searching a library should be able to ask it.
- Return only the question, with no preamble, numbering, or quotation marks.

If the passage is a reference list, a table of raw numbers, an acknowledgements
section, or otherwise has no substantive claim to ask about, reply with exactly:
SKIP

PASSAGE:
{passage}"""


def looks_like_references(text: str) -> bool:
    """Cheap check for bibliography-style passages, before spending an API call."""
    words = len(text.split())
    if words < MIN_WORDS:
        return True
    markers = len(REFERENCEY.findall(text))
    return (markers * 100 / words) > MAX_REFERENCE_RATIO


def write_question(client, passage: str) -> str | None:
    """Ask Claude for one question this passage answers. None if unsuitable."""
    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.format(passage=passage)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text or text.upper().startswith("SKIP"):
        return None
    return text


def main():
    parser = argparse.ArgumentParser(description="Build a retrieval eval test set.")
    parser.add_argument("--n", type=int, default=20, help="How many test cases to build")
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_papers")
    parser.add_argument("--out", default="./evals/testset.jsonl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed so the same chunks are picked each run, "
                             "making results comparable across changes")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (in .env or the environment) first.")

    collection = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    stored = collection.get(limit=collection.count(), include=["documents", "metadatas"])
    pool = list(zip(stored["ids"], stored["documents"], stored["metadatas"]))
    print(f"{len(pool)} chunks available.")

    # Shuffle once with a fixed seed, then walk the list. Same seed -> same
    # test set, so a score change reflects a change you made, not a different
    # random sample.
    random.seed(args.seed)
    random.shuffle(pool)

    client = anthropic.Anthropic()
    cases, skipped_cheap, skipped_model = [], 0, 0

    for chunk_id, document, meta in pool:
        if len(cases) >= args.n:
            break
        if looks_like_references(document):
            skipped_cheap += 1
            continue

        question = write_question(client, document)
        if question is None:
            skipped_model += 1
            continue

        cases.append({
            "question": question,
            "expected_chunk_id": chunk_id,
            "title": meta["title"],
            "arxiv_id": meta.get("arxiv_id", ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            # Kept so a human reviewing the test set can judge the question
            # without going back to the database.
            "passage_preview": " ".join(document.split()[:60]),
        })
        print(f"  [{len(cases)}/{args.n}] {question[:88]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    print(f"\nWrote {len(cases)} test cases to {args.out}")
    print(f"Skipped {skipped_cheap} chunks by the reference-list filter, "
          f"{skipped_model} rejected by the model.")
    if len(cases) < args.n:
        print(f"Only found {len(cases)} of {args.n} requested -- "
              f"the pool ran out of suitable chunks.", file=sys.stderr)


if __name__ == "__main__":
    main()
