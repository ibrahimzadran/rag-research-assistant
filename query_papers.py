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


def retrieve(query: str, collection, vo, top_k=TOP_K):
    result = vo.embed([query], model=EMBED_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "meta": meta})
    return chunks


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
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not voyage_key or not anthropic_key:
        raise SystemExit("Set both VOYAGE_API_KEY and ANTHROPIC_API_KEY as environment variables.")

    vo = voyageai.Client(api_key=voyage_key)
    claude = anthropic.Anthropic(api_key=anthropic_key)
    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.get_collection(args.collection)

    print(f"Retrieving top {args.top_k} chunks for: \"{args.question}\"\n")
    chunks = retrieve(args.question, collection, vo, top_k=args.top_k)

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
