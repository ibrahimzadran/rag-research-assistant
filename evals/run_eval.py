"""
run_eval.py

Scores retrieval quality: for each question, does a chunk that genuinely answers
it come back in the top k?

This measures the FIRST of the three failure modes in a cited RAG system --
whether the right passage is retrieved at all. Nothing downstream can fix a miss
here: Claude cannot cite text it never received.

Two ways to state ground truth, and the file decides which is used:

  expected_chunk_id      A single chunk id. Produced by build_testset.py, which
                         generates a question FROM a chunk. Cheap, but assumes
                         that chunk is the uniquely best answer -- which is often
                         false, so treat its scores as a floor.

  ground_truth_excerpt   A passage you read and copied out of the paper yourself.
                         Any chunk containing it counts as correct, which is both
                         more honest and robust to chunk boundaries: overlapping
                         chunks may each contain the sentence, and all are right.

  notes                  A paraphrase of the answer, written after reading the
                         paper. Cannot be string-matched, so retrieved chunks are
                         judged in rank order until one is found to contain the
                         answer. WEAKER than an excerpt: the label depends on a
                         model's judgement, and an answer sitting in a chunk that
                         was never retrieved cannot be distinguished from no
                         answer existing at all. Requires ANTHROPIC_API_KEY.

Every set also gets a paper-level score: did any chunk from the right paper come
back, and at what rank. That needs no judge and no excerpt, so it is the one
number that is directly comparable across all four sets.

Metrics:

  recall@k  Fraction of questions where a correct chunk appeared in the top k.
  MRR       Mean Reciprocal Rank -- 1.0 if a correct chunk ranked first, 0.5 if
            second, 0 if absent. Recall hides the gap between rank 1 and rank 10;
            MRR does not.

Usage:
    python evals/run_eval.py --dataset evals/manual_qa.jsonl --top_k 10
    python evals/run_eval.py --dataset evals/testset.jsonl   --top_k 10
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from query_papers import RETRIEVE_K, rerank_chunks  # noqa: E402

load_dotenv()

EMBED_MODEL = "voyage-3.5"

# Questions are short (~30 tokens), so all of them fit in a single request well
# under the free tier's 10,000 tokens/minute. Batching means the whole eval costs
# ONE Voyage call rather than one per question, sidestepping the 3-per-minute
# limit entirely. Chroma runs locally, so the searches are free and unmetered.
QUERY_BATCH = 64


def normalise(text):
    """Collapse whitespace and lowercase, so excerpts match despite PDF spacing."""
    return re.sub(r"\s+", " ", text).strip().lower()


def resolve_ground_truth(cases, collection):
    """Return, per case, the set of chunk ids that count as correct.

    Excerpt-based cases are resolved by substring match against every stored
    chunk. A case whose excerpt matches nothing cannot be scored -- usually a
    typo, or text that PDF extraction rendered differently -- so it is reported
    rather than silently counted as a miss, which would look like a retrieval
    failure that never happened.
    """
    stored = collection.get(limit=collection.count(), include=["documents"])
    normalised = [(cid, normalise(doc))
                  for cid, doc in zip(stored["ids"], stored["documents"])]

    resolved, unmatched = [], []
    for case in cases:
        if case.get("expected_chunk_id"):
            resolved.append({case["expected_chunk_id"]})
            continue

        excerpt = normalise(case.get("ground_truth_excerpt", ""))
        if not excerpt:
            unmatched.append((case, "no expected_chunk_id and no ground_truth_excerpt"))
            resolved.append(set())
            continue

        matches = {cid for cid, doc in normalised if excerpt in doc}
        if not matches:
            unmatched.append((case, "excerpt not found in any stored chunk"))
        resolved.append(matches)
    return resolved, unmatched


CONTAINS_PROMPT = """Does the passage below contain the answer described?

QUESTION: {question}

THE ANSWER (as recorded by someone who read the whole paper):
{notes}

PASSAGE:
{passage}

Reply YES only if the passage states the answer, or enough of it that a reader
would learn the answer from this passage alone.

Reply NO in each of these cases, which look like near-misses and are not:

1. The passage only NAMES or MENTIONS the thing asked about without answering
   the question about it. A bibliography entry, a citation, a table of contents,
   a section heading, or a passing reference is a mention, not an answer. If the
   question asks which dataset a paper used, an entry listing that dataset in the
   references does not answer it -- the passage describing its use does.

2. The question has several parts and the passage answers only some. If the
   question asks which of three methods scored highest AND by how much, a passage
   giving the winner's score but not the others' is incomplete. Answer NO.

3. The passage states THAT something is true but the question asks HOW or WHY,
   and the mechanism or reason is absent.

4. Answering would require combining this passage with another one.

Your own knowledge of the subject is irrelevant. Judge only whether THIS passage,
read alone, delivers what the question asks for.

Reply with exactly one word: YES or NO."""


def chunk_contains_answer(claude, question, notes, passage):
    response = claude.messages.create(
        model="claude-sonnet-4-6", max_tokens=8,
        messages=[{"role": "user", "content": CONTAINS_PROMPT.format(
            question=question, notes=notes,
            passage=" ".join(passage.split()[:420]))}])
    text = "".join(b.text for b in response.content if b.type == "text").strip().upper()
    return text.startswith("YES")


def embed_questions(vo, questions):
    """Embed every question, in as few requests as the batch size allows."""
    vectors = []
    for start in range(0, len(questions), QUERY_BATCH):
        batch = questions[start:start + QUERY_BATCH]
        vectors.extend(vo.embed(batch, model=EMBED_MODEL, input_type="query").embeddings)
        if start + QUERY_BATCH < len(questions):
            time.sleep(25)  # only reached with more than 64 questions
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Score retrieval against a question set.")
    parser.add_argument("--dataset", default=None,
                        help="Question set to score (jsonl). Any supported schema.")
    parser.add_argument("--testset", default="./evals/testset.jsonl",
                        help="Deprecated alias for --dataset")
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_papers")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--results_dir", default="./evals/results")
    parser.add_argument("--label", default="")
    parser.add_argument("--rerank", action="store_true",
                        help="Over-retrieve then rerank, as query_papers.py does")
    parser.add_argument("--retrieve_k", type=int, default=RETRIEVE_K)
    parser.add_argument("--rerank_backend", default="local",
                        choices=("local", "voyage"))
    args = parser.parse_args()

    dataset = args.dataset or args.testset
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("Set VOYAGE_API_KEY (in .env or the environment) first.")

    with open(dataset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if not cases:
        raise SystemExit(f"{dataset} is empty -- nothing to score.")

    needs_judge = any(c.get("notes") and not c.get("ground_truth_excerpt")
                      and not c.get("expected_chunk_id") for c in cases)
    if needs_judge and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("This set uses `notes` ground truth, which needs a judge. "
                         "Set ANTHROPIC_API_KEY.")

    collection = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    truth, unmatched = resolve_ground_truth(cases, collection)

    kinds = sorted({("manual" if c.get("ground_truth_excerpt")
                     else "judged" if c.get("notes") else "generated") for c in cases})
    print(f"{os.path.basename(dataset)}: {len(cases)} cases "
          f"({'/'.join(kinds)} ground truth), top_k={args.top_k}\n")

    # Excerpt cases that resolved to nothing cannot be scored; judged cases have
    # no pre-resolvable ids by design, so they are not excluded here.
    unresolvable = {id(c) for c, why in unmatched if not c.get("notes")}
    scorable = [(c, t) for c, t in zip(cases, truth) if id(c) not in unresolvable]
    if unresolvable:
        print(f"WARNING: {len(unresolvable)} case(s) excluded -- excerpt not found "
              f"in any stored chunk:")
        for case, why in unmatched:
            if id(case) in unresolvable:
                print(f"  - {case['question'][:66]}")
        print()

    vo = voyageai.Client()
    vectors = embed_questions(vo, [c["question"] for c, _ in scorable])

    # Fetch more candidates when reranking, then cut back to top_k per question,
    # exactly as query_papers.py does -- the eval must score the shipped path.
    n_fetch = args.retrieve_k if args.rerank else args.top_k
    raw = collection.query(query_embeddings=vectors, n_results=n_fetch,
                           include=["documents", "metadatas"])

    if args.rerank:
        found = {"ids": [], "documents": [], "metadatas": []}
        for (case, _), ids, docs, metas in zip(scorable, raw["ids"],
                                               raw["documents"], raw["metadatas"]):
            cands = [{"id": i, "text": d, "meta": m}
                     for i, d, m in zip(ids, docs, metas)]
            top = rerank_chunks(vo, case["question"], cands, args.top_k,
                                backend=args.rerank_backend)
            found["ids"].append([c["id"] for c in top])
            found["documents"].append([c["text"] for c in top])
            found["metadatas"].append([c["meta"] for c in top])
    else:
        found = raw

    claude = anthropic.Anthropic() if needs_judge else None
    per_case = []
    hits = paper_hits = 0
    rr = paper_rr = 0.0

    for (case, correct_ids), ids, docs, metas in zip(
            scorable, found["ids"], found["documents"], found["metadatas"]):
        if correct_ids:
            # Exact ground truth: best rank among all acceptable chunks.
            ranks = [ids.index(cid) + 1 for cid in correct_ids if cid in ids]
            rank = min(ranks) if ranks else None
            judged_id = None
        else:
            # Judged ground truth: walk the ranking and stop at the first chunk
            # that actually contains the answer. Stopping early keeps the cost
            # near one call per question rather than top_k calls.
            rank, judged_id = None, None
            for i, (doc, cid) in enumerate(zip(docs, ids), 1):
                if chunk_contains_answer(claude, case["question"], case["notes"], doc):
                    rank, judged_id = i, cid
                    break

        # Paper-level: needs no judge and no excerpt, so it is the one measure
        # directly comparable across every set regardless of ground-truth style.
        want_paper = case.get("source_title") or case.get("title")
        paper_ranks = [i for i, m in enumerate(metas, 1) if m.get("title") == want_paper]
        paper_rank = min(paper_ranks) if paper_ranks else None

        if rank:
            hits += 1
            rr += 1 / rank
        if paper_rank:
            paper_hits += 1
            paper_rr += 1 / paper_rank

        per_case.append({"question": case["question"], "rank": rank,
                         "paper_rank": paper_rank, "matched_chunk_id": judged_id,
                         "acceptable_chunk_ids": sorted(correct_ids),
                         "returned": ids})
        chunk_tag = f"rank {rank}" if rank else "MISS"
        paper_tag = f"p{paper_rank}" if paper_rank else "p-MISS"
        print(f"  [{chunk_tag:>6} | {paper_tag:>6}] {case['question'][:64]}")

    n = len(scorable)
    recall, mrr = hits / n, rr / n
    p_recall, p_mrr = paper_hits / n, paper_rr / n

    print(f"\n{'':-<72}")
    print(f"chunk-level   recall@{args.top_k} {recall:.2f}   MRR {mrr:.2f}   ({hits}/{n})")
    print(f"paper-level   recall@{args.top_k} {p_recall:.2f}   MRR {p_mrr:.2f}   ({paper_hits}/{n})")

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "-".join(kinds) + ("-rerank" if args.rerank else "-norerank")
    out_path = os.path.join(args.results_dir, f"retrieval_{tag}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "dataset": dataset, "ground_truth": kinds,
                   "label": args.label, "top_k": args.top_k, "n_cases": n,
                   "rerank": args.rerank,
                   "rerank_backend": args.rerank_backend if args.rerank else None,
                   "retrieve_k": args.retrieve_k if args.rerank else args.top_k,
                   "recall_at_k": recall, "mrr": mrr,
                   "paper_recall_at_k": p_recall, "paper_mrr": p_mrr,
                   "cases": per_case}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
