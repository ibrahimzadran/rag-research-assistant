"""
fetch_arxiv_papers.py

Downloads N papers from arXiv matching a search query and saves them as PDFs,
plus a metadata.jsonl file with title/authors/abstract/source path for each one.

Usage:
    python fetch_arxiv_papers.py --query "retrieval augmented generation" --max 20 --out ./papers

Requires:
    pip install requests feedparser
"""

import argparse
import json
import os
import time
import re
import requests
import feedparser

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def sanitize_filename(title: str) -> str:
    """Turn a paper title into a safe filename."""
    title = re.sub(r"[^\w\s-]", "", title).strip()
    title = re.sub(r"\s+", "_", title)
    return title[:100]  # keep filenames reasonably short


def search_arxiv(query: str, max_results: int):
    """Query the arXiv API and return parsed entries."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return feed.entries


def download_pdf(pdf_url: str, dest_path: str):
    """Stream-download a PDF to disk."""
    with requests.get(pdf_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser(description="Download arXiv papers by topic.")
    parser.add_argument("--query", required=True, help="Search topic, e.g. 'retrieval augmented generation'")
    parser.add_argument("--max", type=int, default=20, help="Number of papers to download")
    parser.add_argument("--out", default="./papers", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metadata_path = os.path.join(args.out, "metadata.jsonl")

    print(f"Searching arXiv for: '{args.query}' (max {args.max} results)...")
    entries = search_arxiv(args.query, args.max)
    print(f"Found {len(entries)} results.")

    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        for i, entry in enumerate(entries, 1):
            title = entry.title.replace("\n", " ").strip()
            arxiv_id = entry.id.split("/abs/")[-1]
            pdf_url = entry.id.replace("/abs/", "/pdf/") + ".pdf"
            authors = [a.name for a in entry.authors]
            abstract = entry.summary.replace("\n", " ").strip()

            filename = f"{sanitize_filename(title)}.pdf"
            dest_path = os.path.join(args.out, filename)

            print(f"[{i}/{len(entries)}] Downloading: {title[:70]}...")
            try:
                download_pdf(pdf_url, dest_path)
            except Exception as e:
                print(f"  Failed to download {arxiv_id}: {e}")
                continue

            record = {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "local_path": dest_path,
            }
            meta_file.write(json.dumps(record) + "\n")

            # Be polite to arXiv's servers
            time.sleep(1)

    print(f"\nDone. Papers saved to '{args.out}/', metadata in '{metadata_path}'.")


if __name__ == "__main__":
    main()
