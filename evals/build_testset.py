"""
build_testset.py

Creates an evaluation test set by working backwards from chunks we already have,
then verifying each case before keeping it.

The generation trick: pick a chunk, ask Claude to write a question that passage
answers, and record the chunk id. Question and ground truth are produced
together, for free.

The trick's flaw, found by inspecting real results: the chunk a question was
generated from is NOT always the best chunk to answer it. Claude will write a
reasonable question from the one substantive paragraph inside an otherwise
bibliographic chunk, and the test set then marks every better passage wrong.
Measured on an unverified set, retrieval was penalised for returning passages
that plainly answered the question better than the "correct" one did.

So every candidate is checked: retrieve for the question, and ask whether the
source chunk really is the best answer among what came back. Cases that fail are
written to evals/discarded_cases.jsonl rather than dropped silently -- the
pattern in those discards tells you which chunk types make bad retrieval targets.

Two passes, deliberately:
  1. Generate all candidate questions   (Anthropic only)
  2. Embed them all in ONE Voyage call, retrieve, and judge

Doing the check inline would need one Voyage call per question, which at the free
tier's 3 requests/minute turns a two-minute job into twenty.

Usage:
    python evals/build_testset.py --n 20
"""

import argparse
import json
import os
import random
import re
import sys

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-3.5"
CHECK_TOP_K = 5

# A passage full of citation years, "et al." and arXiv ids is a reference list.
# This cheap filter saves API calls on the obvious cases; the self-check below
# is what actually catches the mixed ones.
REFERENCEY = re.compile(r"\b(19|20)\d{2}[a-z]?\.|et\s+al\.|arXiv:|CoRR,|preprint",
                        re.IGNORECASE)
MIN_WORDS = 250
MAX_REFERENCE_RATIO = 12

WRITE_PROMPT = """Below is a passage from a research paper.

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

CHECK_PROMPT = """A question was written from PASSAGE A. Other passages were then
retrieved for that same question.

QUESTION: {question}

PASSAGE A (the one the question was written from):
{source}

OTHER RETRIEVED PASSAGES:
{others}

Does PASSAGE A answer the question better than every other passage shown?

Judge only how well each passage answers THIS question. A passage that merely
mentions the topic does not answer it. If PASSAGE A contains the answer only in
passing while another passage explains it directly, PASSAGE A is not the best.

Reply with exactly one word on the first line: BEST or NOT_BEST.
Then one short sentence saying why."""


def looks_like_references(text):
    words = len(text.split())
    if words < MIN_WORDS:
        return True
    return (len(REFERENCEY.findall(text)) * 100 / words) > MAX_REFERENCE_RATIO


def write_question(claude, passage):
    response = claude.messages.create(
        model=CHAT_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": WRITE_PROMPT.format(passage=passage)}])
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return None if not text or text.upper().startswith("SKIP") else text


def is_best_answer(claude, question, source, others):
    """Ask whether the source chunk beats everything else retrieved."""
    block = "\n\n".join(f"[{i}] {' '.join(t.split()[:220])}"
                        for i, t in enumerate(others, 1))
    response = claude.messages.create(
        model=CHAT_MODEL, max_tokens=150,
        messages=[{"role": "user", "content": CHECK_PROMPT.format(
            question=question, source=" ".join(source.split()[:400]), others=block)}])
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    verdict = text.splitlines()[0].strip().upper()
    reason = " ".join(text.splitlines()[1:]).strip()
    return verdict.startswith("BEST"), reason


def main():
    parser = argparse.ArgumentParser(description="Build a verified retrieval eval test set.")
    parser.add_argument("--n", type=int, default=20, help="How many VERIFIED cases to keep")
    parser.add_argument("--oversample", type=float, default=1.6,
                        help="Candidates generated per case wanted, since the "
                             "self-check discards some")
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_papers")
    parser.add_argument("--out", default="./evals/testset.jsonl")
    parser.add_argument("--discards", default="./evals/discarded_cases.jsonl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed so the same chunks are picked each run")
    args = parser.parse_args()

    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"Set {key} (in .env or the environment) first.")

    collection = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    stored = collection.get(limit=collection.count(), include=["documents", "metadatas"])
    pool = list(zip(stored["ids"], stored["documents"], stored["metadatas"]))
    text_by_id = dict(zip(stored["ids"], stored["documents"]))
    print(f"{len(pool)} chunks available.")

    random.seed(args.seed)
    random.shuffle(pool)

    claude = anthropic.Anthropic()
    target = int(args.n * args.oversample)

    # ---- Pass 1: generate candidates (Anthropic only) ----
    print(f"\nPass 1: generating up to {target} candidate questions...")
    candidates, skipped_cheap, skipped_model = [], 0, 0
    for chunk_id, document, meta in pool:
        if len(candidates) >= target:
            break
        if looks_like_references(document):
            skipped_cheap += 1
            continue
        question = write_question(claude, document)
        if question is None:
            skipped_model += 1
            continue
        candidates.append({"question": question, "expected_chunk_id": chunk_id,
                           "title": meta["title"], "arxiv_id": meta.get("arxiv_id", ""),
                           "page_start": meta.get("page_start"),
                           "page_end": meta.get("page_end"),
                           "passage_preview": " ".join(document.split()[:60])})
    print(f"  {len(candidates)} candidates "
          f"({skipped_cheap} filtered as bibliography, {skipped_model} refused by the model)")

    # ---- Pass 2: verify each candidate (one Voyage call, then judges) ----
    print(f"\nPass 2: checking whether each source chunk is really the best answer...")
    vectors = voyageai.Client().embed([c["question"] for c in candidates],
                                      model=EMBED_MODEL, input_type="query").embeddings
    retrieved = collection.query(query_embeddings=vectors, n_results=CHECK_TOP_K)

    kept, discarded = [], []
    for cand, ids, docs in zip(candidates, retrieved["ids"], retrieved["documents"]):
        others = [(i, d) for i, d in zip(ids, docs) if i != cand["expected_chunk_id"]]
        if not others:
            kept.append(cand)  # nothing else came back to compare against
            continue

        ok, reason = is_best_answer(claude, cand["question"],
                                    text_by_id[cand["expected_chunk_id"]],
                                    [d for _, d in others])
        if ok:
            kept.append(cand)
            print(f"  KEEP    {cand['question'][:70]}")
        else:
            discarded.append({**cand, "discard_reason": reason,
                              "beaten_by": [i for i, _ in others],
                              "beaten_by_preview": [" ".join(d.split()[:40])
                                                    for _, d in others[:2]]})
            print(f"  DISCARD {cand['question'][:70]}")
            print(f"          -> {reason[:92]}")
        if len(kept) >= args.n:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for case in kept:
            f.write(json.dumps(case) + "\n")
    with open(args.discards, "w", encoding="utf-8") as f:
        for case in discarded:
            f.write(json.dumps(case) + "\n")

    checked = len(kept) + len(discarded)
    print(f"\n{'':-<72}")
    print(f"kept ......... {len(kept)} -> {args.out}")
    print(f"discarded .... {len(discarded)} -> {args.discards}")
    if checked:
        print(f"discard rate . {100 * len(discarded) // checked}% of candidates checked")
    if len(kept) < args.n:
        print(f"\nOnly {len(kept)} of {args.n} requested survived. Raise --oversample "
              f"or --n to generate more candidates.", file=sys.stderr)


if __name__ == "__main__":
    main()
