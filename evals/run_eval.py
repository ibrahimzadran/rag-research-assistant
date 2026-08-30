"""
run_eval.py

Scores retrieval quality: for each question in the test set, does the chunk we
know is correct actually come back in the top-k results?

This measures the FIRST of the three failure modes in a cited RAG system --
whether the right passage is retrieved at all. Nothing downstream can fix a
miss here: Claude cannot cite text it never received.

Two numbers are reported:

  recall@k  The fraction of questions where the correct chunk appeared anywhere
            in the top k. "Did we find it?"

  MRR       Mean Reciprocal Rank. Scores 1.0 if the correct chunk ranked first,
            0.5 if second, 0.33 if third, 0 if absent. "Did we rank it well?"
            Recall alone hides the difference between rank 1 and rank 5; MRR
            does not, and rank matters because the answer model weighs earlier
            sources more heavily.

Results are written to evals/results/ so two runs can be compared after a
change to chunk size, overlap, or top_k.

Usage:
    python evals/run_eval.py --top_k 5
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "voyage-3.5"

# Questions are short (~30 tokens), so all of them fit inside a single request
# well under the free tier's 10,000 tokens/minute. Batching this way means the
# whole eval costs ONE Voyage call rather than one per question -- which also
# sidesteps the 3-requests-per-minute limit entirely. Chroma itself runs
# locally, so the searches are free and unmetered.
QUERY_BATCH = 64


def embed_questions(vo, questions):
    """Embed every question, in as few requests as the batch size allows."""
    vectors = []
    for start in range(0, len(questions), QUERY_BATCH):
        batch = questions[start:start + QUERY_BATCH]
        result = vo.embed(batch, model=EMBED_MODEL, input_type="query")
        vectors.extend(result.embeddings)
        if start + QUERY_BATCH < len(questions):
            time.sleep(25)  # only reached with >64 questions
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Score retrieval against the test set.")
    parser.add_argument("--testset", default="./evals/testset.jsonl")
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_papers")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Must match what query_papers.py uses, or the score "
                             "will not describe the system you actually run")
    parser.add_argument("--results_dir", default="./evals/results")
    parser.add_argument("--label", default="",
                        help="Note describing this run, e.g. 'chunk500-overlap75'")
    args = parser.parse_args()

    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("Set VOYAGE_API_KEY (in .env or the environment) first.")

    with open(args.testset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(cases)} test cases. Retrieving top {args.top_k} for each.\n")

    collection = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    vo = voyageai.Client()

    vectors = embed_questions(vo, [c["question"] for c in cases])

    # One local search per question -- no API, no rate limit.
    found = collection.query(query_embeddings=vectors, n_results=args.top_k)

    per_case, hits, reciprocal_total = [], 0, 0.0
    for case, returned_ids in zip(cases, found["ids"]):
        expected = case["expected_chunk_id"]
        # rank is 1-based; None means the correct chunk never came back.
        rank = returned_ids.index(expected) + 1 if expected in returned_ids else None
        if rank:
            hits += 1
            reciprocal_total += 1 / rank
        per_case.append({
            "question": case["question"],
            "expected_chunk_id": expected,
            "title": case["title"],
            "rank": rank,
            "returned": returned_ids,
        })
        marker = f"rank {rank}" if rank else "MISS"
        print(f"  [{marker:>6}] {case['question'][:78]}")

    recall = hits / len(cases)
    mrr = reciprocal_total / len(cases)

    print(f"\n{'':-<72}")
    print(f"recall@{args.top_k} .... {recall:.2f}  ({hits} of {len(cases)} found)")
    print(f"MRR ........... {mrr:.2f}  (1.00 = always ranked first)")

    misses = [c for c in per_case if c["rank"] is None]
    if misses:
        print(f"\n{len(misses)} miss(es) -- these are where retrieval failed:")
        for m in misses:
            print(f"  - {m['question'][:70]}")
            print(f"      wanted: {m['expected_chunk_id']}")

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.results_dir, f"{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": stamp,
            "label": args.label,
            "top_k": args.top_k,
            "n_cases": len(cases),
            "recall_at_k": recall,
            "mrr": mrr,
            "cases": per_case,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
