"""
query_papers.py

Queries the Chroma vector store built by ingest_papers.py, retrieves the
top-k most relevant chunks, and asks Claude to answer using only those
chunks, with numbered citations back to the source paper + page.

Usage:
    python query_papers.py --question "How does Ragas evaluate RAG systems?"

Requires:
    pip install chromadb voyageai anthropic

Environment variables needed:
    export VOYAGE_API_KEY="your-voyage-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
"""

import argparse
import os
import time

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv

# Read key=value pairs from a .env file in the current directory into the
# environment, so os.environ.get() below finds them. Real shell environment
# variables still win, so exporting a key manually overrides the file.
load_dotenv()

EMBED_MODEL = "voyage-3.5"
CHAT_MODEL = "claude-sonnet-4-6"
# Raised from 5 to 10 on the basis of evals/run_eval.py: on the 20-question
# test set, recall@5 was 0.90 (2 of 20 correct passages never retrieved)
# while recall@10 was 1.00. Going beyond 10 added nothing but tokens.
TOP_K = 10

# Over-retrieve then rerank. Embedding similarity scores a chunk as a whole, so a
# single decisive sentence inside a 500-word chunk about something else gets
# averaged away -- measured case: the chunk answering "why is directly training a
# model impractical" ranks 16th of 460, because its answer starts at word 113 and
# the chunk opens with unrelated complexity notation. A reranker over the top 10
# cannot fix that; the chunk is not in the top 10. Fetching 50 first and reranking
# down to 10 can, because a cross-encoder scores the query against the passage
# jointly rather than comparing two independent summaries of meaning.
RETRIEVE_K = 50

# Two reranker backends, same job, very different constraints.
#
#   "local"   A cross-encoder run on this machine. Free, unmetered, offline, and
#             fast enough that reranking stops being a scheduling problem. Needs
#             sentence-transformers (pulls in torch, ~2.5GB) and downloads the
#             model once on first use. This is the default because Voyage's free
#             tier cannot sustain reranking -- see below.
#
#   "voyage"  Voyage's hosted rerank-2.5. Better model, no local dependency, but
#             on the no-payment-method tier a single request may hold at most
#             ~8 chunks and requests must be a full minute apart. That is ~8
#             minutes per query at RETRIEVE_K=50, which makes evaluating it
#             across a question set impractical.
#
# Either way the query is still embedded through Voyage, so an API key is
# required regardless; only the reranking step moves.
RERANK_BACKEND = "local"
# bge-reranker-base and bge-reranker-v2-m3 both fail to load under
# transformers 5.x -- their checkpoints predate the format it expects. This
# MiniLM cross-encoder loads cleanly, is ~90MB rather than ~1.1GB, and separates
# relevant from irrelevant passages by a wide margin in a smoke test.
LOCAL_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VOYAGE_RERANK_MODEL = "rerank-2.5"

# Reranking 50 full chunks is ~33,000 tokens in one request, which exceeds the
# free tier's cap outright -- it can never succeed. Batching is the only fix that
# keeps the point of the change: truncating chunks to fit would hide exactly the
# late-in-chunk content the reranker exists to surface.
#
# Measured, not assumed: 8 chunks (~5,300 tokens) succeeds, 12 (~8,000) does not,
# even with 65s between requests -- so the effective ceiling is well below the
# documented 10,000 tokens/minute. With a payment method, raise RERANK_BATCH to
# cover all candidates in one request and set the pause to 0.
RERANK_BATCH = 8
# Two 5,300-token requests inside one minute is 10,600 tokens, over the cap, so
# the spacing has to clear a full minute. That makes reranking ~1 request/minute
# on the free tier: 7 batches for 50 candidates is ~8 minutes per query.
RERANK_PAUSE_SECONDS = 68


def retrieve(query: str, collection, vo, top_k=TOP_K, rerank=True,
             retrieve_k=RETRIEVE_K, backend=None):
    """Fetch candidates by embedding similarity, optionally rerank, return top_k.

    With rerank=False this is the original behaviour: embed, take top_k, done.
    Kept so the two paths can be compared on the same question set.
    """
    result = vo.embed([query], model=EMBED_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    n_candidates = retrieve_k if rerank else top_k
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
    )

    chunks = []
    for cid, doc, meta in zip(results["ids"][0], results["documents"][0],
                              results["metadatas"][0]):
        chunks.append({"id": cid, "text": doc, "meta": meta})

    if not rerank:
        return chunks

    return rerank_chunks(vo, query, chunks, top_k, backend=backend)


_LOCAL_RERANKER = None


def _local_reranker():
    """Load the cross-encoder once and keep it. Imported lazily so the project
    still runs without sentence-transformers when reranking is off or remote."""
    global _LOCAL_RERANKER
    if _LOCAL_RERANKER is None:
        from sentence_transformers import CrossEncoder
        _LOCAL_RERANKER = CrossEncoder(LOCAL_RERANK_MODEL)
    return _LOCAL_RERANKER


def rerank_chunks(vo, query, chunks, top_k, backend=None):
    """Reorder candidates by relevance to the query and keep the best top_k.

    A cross-encoder reads the question and the passage together, rather than
    comparing two independently-produced summaries of meaning. That is what lets
    it notice an answer occupying one sentence of a 500-word chunk, which is
    precisely what embedding similarity averages away.
    """
    backend = backend or RERANK_BACKEND
    if backend == "local":
        model = _local_reranker()
        scores = model.predict([(query, c["text"]) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:top_k]]
    if backend != "voyage":
        raise ValueError(f"unknown rerank backend {backend!r}; use 'local' or 'voyage'")
    return _rerank_voyage(vo, query, chunks, top_k)


def _rerank_voyage(vo, query, chunks, top_k):
    """Rerank via Voyage, in batches small enough for the free tier.

    Scores are query-document relevance from one model, so they are comparable
    across batches and a global sort is sound. With a payment method, raise
    RERANK_BATCH to cover all candidates in one request and set the pause to 0.
    """
    scored = []
    for start in range(0, len(chunks), RERANK_BATCH):
        batch = chunks[start:start + RERANK_BATCH]
        if start:
            time.sleep(RERANK_PAUSE_SECONDS)
        ranked = vo.rerank(query=query, documents=[c["text"] for c in batch],
                           model=VOYAGE_RERANK_MODEL)
        for r in ranked.results:
            # `index` refers back into the batch we passed in.
            scored.append((r.relevance_score, batch[r.index]))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def format_pages(meta: dict) -> str:
    """Render a chunk's page span, e.g. "page 7" or "pages 7-8".

    Most chunks straddle a page boundary (78% of this corpus), so a single
    page number would often point at the wrong page. Falls back to the older
    single-page metadata for anything ingested before page spans were stored.
    """
    start = meta.get("page_start")
    end = meta.get("page_end")
    if start is None or end is None:
        return f"page {meta.get('page', '?')}"
    return f"page {start}" if start == end else f"pages {start}-{end}"


def build_prompt(question: str, chunks: list):
    sources_block = ""
    for i, c in enumerate(chunks, 1):
        title = c["meta"].get("title", "unknown")
        pages = format_pages(c["meta"])
        sources_block += f"[{i}] (Source: {title}, {pages})\n{c['text']}\n\n"

    prompt = f"""Answer the question using ONLY the sources below. If the sources don't contain
enough information to answer, say so explicitly rather than guessing.

Cite every factual claim with the matching bracketed number, e.g. [1], [2].
Multiple sources can support one claim, e.g. [1][3].
Cite each bullet point on its own line rather than relying on a citation in an
introductory line above it.

Reproduce names exactly as the sources write them - method names, module names,
metric names, dataset names, and acronyms. Do not paraphrase them, and do not
combine words from two different names into one. If you are not certain of a
name's exact wording, quote the phrase the source uses.

SOURCES:
{sources_block}
QUESTION: {question}

ANSWER:"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Ask a cited question over your ingested papers.")
    parser.add_argument("--question", required=True, help="Your question")
    parser.add_argument("--db_dir", default="./chroma_db", help="Chroma DB directory")
    parser.add_argument("--collection", default="rag_papers", help="Chroma collection name")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="Chunks sent to the answer model")
    parser.add_argument("--retrieve_k", type=int, default=RETRIEVE_K,
                        help="Candidates fetched before reranking")
    parser.add_argument("--rerank_backend", default=RERANK_BACKEND,
                        choices=("local", "voyage"),
                        help="Where reranking runs")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Skip reranking and use plain embedding order "
                             "(the pre-reranker behaviour, for A/B comparison)")
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not voyage_key or not anthropic_key:
        raise SystemExit("Set both VOYAGE_API_KEY and ANTHROPIC_API_KEY as environment variables.")

    vo = voyageai.Client(api_key=voyage_key)
    claude = anthropic.Anthropic(api_key=anthropic_key)
    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.get_collection(args.collection)

    how = (f"retrieving {args.retrieve_k}, reranking ({args.rerank_backend}) to {args.top_k}"
           if args.rerank else f"retrieving top {args.top_k}, no rerank")
    print(f"{how} for: \"{args.question}\"\n")
    chunks = retrieve(args.question, collection, vo, top_k=args.top_k,
                      rerank=args.rerank, retrieve_k=args.retrieve_k,
                      backend=args.rerank_backend)

    print("--- Retrieved sources ---")
    for i, c in enumerate(chunks, 1):
        title = c["meta"].get("title", "unknown")
        print(f"[{i}] {title} ({format_pages(c['meta'])})")
    print()

    prompt = build_prompt(args.question, chunks)
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_text = "".join(block.text for block in response.content if block.type == "text")

    print("--- Answer ---")
    print(answer_text)
    print("\n--- Source key ---")
    for i, c in enumerate(chunks, 1):
        title = c["meta"].get("title", "unknown")
        arxiv_id = c["meta"].get("arxiv_id", "")
        print(f"[{i}] {title} (arXiv:{arxiv_id}, {format_pages(c['meta'])})")


if __name__ == "__main__":
    main()
