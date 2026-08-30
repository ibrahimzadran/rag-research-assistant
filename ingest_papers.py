"""
ingest_papers.py

Extracts text from PDFs, splits it into overlapping chunks, embeds each chunk
with Voyage AI, and stores everything in a local persistent Chroma collection.

Usage:
    python ingest_papers.py --papers_dir ./papers --db_dir ./chroma_db

Requires:
    pip install pypdf chromadb voyageai

Set your Voyage API key first:
    export VOYAGE_API_KEY="your-key-here"
    (get one free at https://dashboard.voyageai.com/)
"""

import argparse
import glob
import json
import os
import time

import chromadb
import voyageai
from dotenv import load_dotenv
from pypdf import PdfReader

# Read key=value pairs from a .env file in the current directory into the
# environment, so os.environ.get() below finds them. Real shell environment
# variables still win, so exporting a key manually overrides the file.
load_dotenv()

EMBED_MODEL = "voyage-3.5"  # good general-purpose Voyage embedding model
CHUNK_SIZE = 500       # approx words per chunk
CHUNK_OVERLAP = 75     # words of overlap between consecutive chunks

# Voyage's no-payment-method tier allows 3 requests/min and 10,000 tokens/min.
# The binding constraint is tokens, not requests: a 500-word chunk is roughly
# 670 tokens, so 6 chunks is ~4,000 tokens and two requests per minute stays
# near 8,000 -- safely under the cap. (Measured: batches up to 8 succeed when
# spaced a minute apart. The original 32-chunk batch was ~21,000 tokens and
# exceeded the per-minute cap in a single request, so it could never succeed.)
# With a payment method the limits lift; raise the batch and set the pause to 0.
EMBED_BATCH_SIZE = 6
EMBED_PAUSE_SECONDS = 32
EMBED_MAX_ATTEMPTS = 5


def extract_text_by_page(pdf_path: str):
    """Return a list of (page_number, text) tuples for a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Concatenate all page text with page markers, then split into
    overlapping word-based chunks. Returns list of dicts with text + page.
    """
    words_with_pages = []
    for page_num, text in pages:
        for word in text.split():
            words_with_pages.append((word, page_num))

    chunks = []
    start = 0
    while start < len(words_with_pages):
        end = min(start + chunk_size, len(words_with_pages))
        window = words_with_pages[start:end]
        chunk_text = " ".join(w for w, _ in window)
        page_counts = {}
        for _, p in window:
            page_counts[p] = page_counts.get(p, 0) + 1
        # `page` is the single best-guess page (the one contributing the most
        # words). It is only reliable for chunks that sit inside one page --
        # measured on this corpus, 78% of chunks span a page boundary, because
        # pages average ~482 words against a 500-word chunk. page_start/page_end
        # record the true span so a citation can say "pages 7-8" instead of
        # guessing 7 when the sentence is on 8.
        chunk_page = max(page_counts, key=page_counts.get)
        pages_in_chunk = [p for _, p in window]
        chunks.append({
            "text": chunk_text,
            "page": chunk_page,
            "page_start": min(pages_in_chunk),
            "page_end": max(pages_in_chunk),
        })
        if end == len(words_with_pages):
            break
        start += chunk_size - overlap
    return chunks


def embed_adaptive(vo, texts):
    """Embed one batch, halving it whenever Voyage refuses. Returns embeddings.

    On the free tier a refusal is routine, not exceptional. Crucially, a refused
    request still appears to consume the per-minute token budget, so retrying
    the *same* size can never recover -- each attempt refills the very budget it
    is waiting on. Two earlier runs stalled that way, one after 190s of backoff
    and one after 440s.

    Halving escapes that trap: if 6 chunks will not fit, 3 might, and 1 almost
    always will. The batch size self-tunes to whatever headroom exists instead
    of relying on constants guessed in advance. Only when a single chunk is
    refused is waiting the sole remaining option.
    """
    try:
        return vo.embed(texts, model=EMBED_MODEL,
                        input_type="document").embeddings
    except voyageai.error.RateLimitError:
        if len(texts) > 1:
            mid = len(texts) // 2
            print(f"    refused at {len(texts)} chunks - splitting into "
                  f"{mid} + {len(texts) - mid}", flush=True)
            time.sleep(EMBED_PAUSE_SECONDS)
            first = embed_adaptive(vo, texts[:mid])
            time.sleep(EMBED_PAUSE_SECONDS)
            return first + embed_adaptive(vo, texts[mid:])

        # A single chunk was refused -- nothing left to split, so wait it out.
        for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
            wait = 65 + 30 * (attempt - 1)
            print(f"    single chunk refused - waiting {wait}s "
                  f"(attempt {attempt}/{EMBED_MAX_ATTEMPTS})", flush=True)
            time.sleep(wait)
            try:
                return vo.embed(texts, model=EMBED_MODEL,
                                input_type="document").embeddings
            except voyageai.error.RateLimitError:
                continue
        raise


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into a Chroma vector store.")
    parser.add_argument("--papers_dir", default="./papers", help="Folder containing PDFs + metadata.jsonl")
    parser.add_argument("--db_dir", default="./chroma_db", help="Where to persist the Chroma DB")
    parser.add_argument("--collection", default="rag_papers", help="Chroma collection name")
    args = parser.parse_args()

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("Set VOYAGE_API_KEY as an environment variable first.")

    vo = voyageai.Client(api_key=api_key)
    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.get_or_create_collection(args.collection)

    metadata_path = os.path.join(args.papers_dir, "metadata.jsonl")
    metadata_by_path = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                metadata_by_path[record["local_path"]] = record

    pdf_paths = sorted(glob.glob(os.path.join(args.papers_dir, "*.pdf")))
    print(f"Found {len(pdf_paths)} PDFs to ingest.")

    # Resume support: chunk ids are deterministic ("Paper.pdf::chunk7"), so
    # anything already in the collection can be skipped. This makes the run
    # safe to interrupt with Ctrl+C and restart later -- useful on the free
    # tier, where a full pass takes ~30 minutes of mostly waiting.
    already_done = set(collection.get(include=[])["ids"])
    if already_done:
        print(f"Resuming - {len(already_done)} chunks already embedded, skipping those.")

    embedded_now = 0
    requests_made = 0
    for pdf_path in pdf_paths:
        record = metadata_by_path.get(pdf_path, {})
        title = record.get("title", os.path.basename(pdf_path))

        pages = extract_text_by_page(pdf_path)
        if not pages:
            print(f"Processing: {title[:70]}\n  No extractable text, skipping.")
            continue

        chunks = chunk_pages(pages)
        base = os.path.basename(pdf_path)

        # Pair every chunk with its id up front, then drop the ones already stored.
        pending = [(f"{base}::chunk{i}", c) for i, c in enumerate(chunks)
                   if f"{base}::chunk{i}" not in already_done]
        if not pending:
            print(f"Done already: {title[:60]} ({len(chunks)} chunks)")
            continue

        print(f"Processing: {title[:70]}")
        print(f"  {len(chunks)} chunks, {len(pending)} still to embed")

        for batch_start in range(0, len(pending), EMBED_BATCH_SIZE):
            batch = pending[batch_start:batch_start + EMBED_BATCH_SIZE]
            ids = [i for i, _ in batch]
            texts = [c["text"] for _, c in batch]

            # Space every request except the first of the session. Done here
            # rather than after the batch so the gap also applies when moving
            # from one paper to the next -- the rate limit is account-wide and
            # does not reset at file boundaries.
            if requests_made:
                time.sleep(EMBED_PAUSE_SECONDS)
            embeddings = embed_adaptive(vo, texts)
            requests_made += 1

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "title": title,
                        "arxiv_id": record.get("arxiv_id", ""),
                        "source_path": pdf_path,
                        "page": c["page"],
                        "page_start": c["page_start"],
                        "page_end": c["page_end"],
                    }
                    for _, c in batch
                ],
            )
            already_done.update(ids)
            embedded_now += len(batch)
            print(f"    stored {len(batch)} chunks "
                  f"({embedded_now} this session)", flush=True)

    total_chunks = embedded_now

    print(f"\nDone. Ingested {total_chunks} chunks into Chroma collection "
          f"'{args.collection}' at '{args.db_dir}'.")
    print(f"Collection now holds {collection.count()} chunks in total.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Everything embedded so far is already committed to Chroma, and chunk
        # ids are deterministic, so re-running resumes rather than duplicating.
        print("\n\nStopped. Progress is saved -- re-run the same command to "
              "carry on where this left off.")
