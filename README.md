# rag-research-assistant

Ask a research question, get an answer built only from papers you have on disk,
with a citation to the paper and page behind every claim.

The point is that nothing is invented: each statement traces to a passage you
can open and check.

## How it works

Three stages, three scripts:

| Stage | Script | What it does |
|---|---|---|
| Fetch | `fetch_arxiv_papers.py` | Downloads papers from arXiv, records title/authors/id in `papers/metadata.jsonl` |
| Ingest | `ingest_papers.py` | Extracts PDF text, splits into overlapping chunks, embeds them, stores in Chroma |
| Query | `query_papers.py` | Embeds your question, retrieves the closest chunks, has Claude answer from them |

Embeddings (text turned into numbers so similar meanings sit close together)
come from Voyage AI. Answers come from Claude. The vector store is Chroma,
which runs locally — searches cost nothing and are not rate limited.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env      # then put real keys in .env, NOT in .env.example
```

`.env` is gitignored; `.env.example` is committed and must only ever hold
placeholders. Both scripts read `.env` automatically — no need to export
anything.

Keys needed: [Voyage AI](https://dashboard.voyageai.com/) for embeddings,
[Anthropic](https://console.anthropic.com/settings/keys) for answers.

## Usage

```bash
# 1. Download papers (~47 MB for 20 papers; PDFs are gitignored)
./venv/bin/python fetch_arxiv_papers.py --query "retrieval augmented generation" --max 20

# 2. Build the searchable database (see the rate-limit note below)
./venv/bin/python ingest_papers.py

# 3. Ask something
./venv/bin/python query_papers.py --question "How does Ragas evaluate RAG systems?"
```

## Evaluation

Three things fail independently, so they are measured separately. A single
score cannot tell them apart.

```bash
./venv/bin/python evals/build_testset.py --n 20   # generate questions + ground truth
./venv/bin/python evals/run_eval.py --top_k 10    # retrieval quality
./venv/bin/python evals/eval_citations.py --n 5   # are cited claims actually supported
```

Results land in `evals/results/` with a timestamp, so two runs can be compared
after a change.

### Measured baseline (20 papers, 460 chunks)

| Metric | Score | Meaning |
|---|---|---|
| recall@10 | 1.00 | The correct passage is always retrieved |
| recall@5 | 0.90 | Two of twenty missed — why `TOP_K` is 10 |
| MRR | 0.67 | Correct passage ranked first about half the time |
| Citation accuracy | 0.92 | Cited claims genuinely supported by their source |
| Citation coverage | 1.00 | Every substantive line carries a citation |

**Read these with suspicion.** Test questions are generated *from* the passages
they target, so they reuse that passage's vocabulary. Real questions, phrased
in your own words, are harder. Treat recall as an upper bound, not a promise.
Sample sizes are also small — 20 retrieval cases, ~49 claims — so small
differences between runs are noise, not signal.

## Things that will bite you

**Voyage rate limits without a payment method.** 3 requests/minute and 10,000
tokens/minute. A full ingest of 20 papers is ~460 chunks and takes roughly
40 minutes under those limits. Two things make that survivable:

- **Ingest resumes.** Chunk ids are deterministic, so `Ctrl+C` and re-run picks
  up where it stopped. Nothing is re-embedded.
- **Batches halve on refusal.** A refused request still consumes the per-minute
  budget, so retrying at the same size never recovers. Halving (6 → 3 → 1)
  guarantees progress.

With a payment method the limits lift: raise `EMBED_BATCH_SIZE` and set
`EMBED_PAUSE_SECONDS = 0` in `ingest_papers.py`.

**Rebuilding the database costs that 40 minutes.** `chroma_db/` is gitignored
because it is a generated binary store. Keep a copy outside git:
`cp -r chroma_db chroma_db.backup`.

**Path matching between stages.** `ingest_papers.py` joins PDFs to their
metadata by exact path string. Run it with `--papers_dir papers` instead of
`./papers` and every lookup silently misses — no error, but citations lose
their titles and arXiv ids. Stick to the defaults.

## Design notes

**Chunks carry a page range, not a page number.** Pages in this corpus average
482 words against a 500-word chunk, so 77% of chunks straddle a page boundary.
Labelling such a chunk with one page cites the wrong page whenever the relevant
sentence falls in the smaller half. Citations read `pages 5-6` instead.

**The overlap is 75 words.** Any span shorter than that appears intact in at
least one chunk, so a sentence split by a chunk boundary still exists whole
somewhere. The cost is ~18% more chunks and some near-duplicate retrieval.

**The eval imports the real prompt.** `eval_citations.py` imports `build_prompt`
from `query_papers.py` rather than copying it. An eval with its own copy scores
a system you do not ship.

## Known limitations

- Test questions share vocabulary with their source passages (see above).
- `TOP_K = 10` retrieves more context than most answers use — a reranker would
  recover the waste.
- The citation judge is an LLM. Two of its verdicts were hand-checked against
  the PDFs and both were correct, but that is a sample of two.
- Claim extraction is line-based. Bullets under an introductory line that holds
  the citation are handled, but unusual formatting may not be.
