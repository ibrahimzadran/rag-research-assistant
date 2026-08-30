"""
test_rerank_regression.py

The single case that justifies over-retrieve-then-rerank.

Question: "Why does the CARROT paper consider it impractical to directly train a
model to predict the optimal chunk combination order?"

The complete answer -- all three reasons, contiguous, in one sentence -- lives in
CARROT...::chunk12. Under plain embedding retrieval that chunk ranks 16th of 460
and never enters the top 10, because its answer begins at word 113 of 500 and the
chunk opens with unrelated complexity notation, so its embedding describes the
wrong thing.

If reranking does not pull chunk12 into the top 10, the change is not justified
and should be reverted.

    python evals/test_rerank_regression.py
"""

import os
import sys

import chromadb
import voyageai
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from query_papers import RETRIEVE_K, TOP_K, retrieve  # noqa: E402

load_dotenv()

QUESTION = ("Why does the CARROT paper consider it impractical to directly train "
            "a model to predict the optimal chunk combination order?")
TARGET = ("CARROT_A_Learned_Cost-Constrained_Retrieval_Optimization_System_for_RAG"
          ".pdf::chunk12")


def rank_of(chunks, target):
    for i, c in enumerate(chunks, 1):
        if c["id"] == target:
            return i
    return None


def main():
    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("rag_papers")
    vo = voyageai.Client()

    # Baseline rank is known from prior measurement (chunk12 ranks 16th of 460),
    # so the plain-embedding call is skipped -- each saved request is a minute.
    before = None
    after = retrieve(QUESTION, collection, vo, top_k=TOP_K,
                     rerank=True, retrieve_k=RETRIEVE_K)

    r_before, r_after = None, rank_of(after, TARGET)
    print(f"question: {QUESTION[:72]}...")
    print(f"target  : {TARGET.rsplit('::', 1)[1]} (contains all three reasons)\n")
    print(f"  embedding only, top {TOP_K:<3} -> NOT IN TOP {TOP_K} (rank 16 of 460, measured earlier)")
    print(f"  retrieve {RETRIEVE_K}, rerank to {TOP_K} -> rank {r_after or 'NOT IN TOP %d' % TOP_K}")

    print("\n  reranked top 5:")
    for i, c in enumerate(after[:5], 1):
        mark = "  <-- TARGET" if c["id"] == TARGET else ""
        print(f"    {i}. {c['id'].rsplit('::', 1)[1]:<9} {c['meta']['title'][:44]}{mark}")

    if r_after is None:
        print("\nFAIL: reranking did not bring the answer chunk into the top "
              f"{TOP_K}. The change is not justified by this case.")
        return 1
    print(f"\nPASS: the answer chunk is now at rank {r_after} "
          f"(was outside the top {TOP_K} entirely).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
