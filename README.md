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

Metrics are listed by how much they can be trusted, which is not the same as how
impressive they look.

**Primary — paper-level retrieval.** Did any chunk from the right paper come
back, and at what rank. This needs no judge and no hand-copied excerpt, so it is
the only figure directly comparable across every question set.

| Metric | Score |
|---|---|
| Paper-level recall@10 | **1.00** |
| Paper-level MRR | **1.00** |

The correct paper ranks first for every question tried, whether the question was
machine-generated or hand-written after reading the paper.

**Secondary — chunk-level retrieval. Judge-dependent estimates, not measurements.**
Deciding whether a specific chunk answers a question requires either an exact
excerpt or a model's judgement, and both are fallible.

| Question set | n | Ground truth | recall@10 | MRR |
|---|---|---|---|---|
| Generated, unverified | 20 | one chunk id | 1.00 | 0.67 |
| Generated, self-checked | 20 | one chunk id | 0.95 | 0.71 |
| Hand-written, full-text read | 13 | LLM-judged | 0.92 | **0.75** (corrected) |

The hand-written set was first reported at MRR 0.88. All 13 cases were then
verified by hand against the PDFs: **2 of 12 judged calls were wrong** (17%). In
one, the judge accepted a bibliography entry naming a dataset as an answer to
"which dataset did this paper use"; the real answer was at rank 8. In another it
accepted an abstract containing one of three numbers a comparison question
needed; the real answer was at rank 6. Correcting both gives 0.75, or 0.73 if a
third borderline case is also counted against.

Generated sets carry the opposite bias: they demand one *specific* chunk, so a
different chunk that answers perfectly scores as a miss. Their figures are
therefore floors. Read every chunk-level number as an estimate with roughly
±0.05 of judge noise, and prefer the paper-level row when comparing anything.

**Answer quality (5 questions, ~49 claims):**

| Metric | Score |
|---|---|
| Citation accuracy | 0.92 |
| Citation coverage | 1.00 |

Sample sizes throughout are small -- 13 to 20 questions, ~49 claims. Differences
smaller than about 0.05 are noise, not signal.

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

**Embedding dilution: a decisive sentence can be buried by its own chunk.** A
chunk's vector represents its 500 words as a whole, so a single sentence about a
different topic gets averaged away and the chunk ranks far lower than its content
deserves.

Documented example. For the question *"why does the CARROT paper consider it
impractical to directly train a model to predict the optimal chunk combination
order?"*, the complete answer is one sentence in `CARROT...::chunk12`:

> "However, this is impractical: (i) the N! combinatorial space makes supervised
> data collection infeasible; (ii) non-monotonic utility functions ... are
> difficult for neural networks to approximate; (iii) predicting chunk
> combination orders requires structured outputs from the N! space, whereas
> predicting MCTS hyperparameters reduces to lightweight regression."

All three reasons, contiguous, in one chunk. Yet that chunk ranks **16th of 460**
and never enters the top 10. The reason is positional: the answer begins at word
113 of 500, and the chunk opens with space-complexity notation and a section on
the configuration agent. Its embedding describes the configuration agent, not the
argument buried inside it.

This is the single confirmed retrieval failure in the hand-written set, and it
constrains the fix. A reranker over the current top 10 would not help, because
the chunk is not in the top 10. The pattern that would work is **over-retrieve
then rerank** -- fetch ~50, rerank, keep 10. Rank 16 sits comfortably inside 50.

Other limitations:

- Generated test questions share vocabulary with their source passages, and
  demand one specific chunk when several may answer equally well.
- The relevance and citation judges are LLMs. Both have been hand-checked
  (2 of 2 citation verdicts correct; 10 of 12 relevance verdicts correct), but
  neither is validated at scale.
- `TOP_K = 10` retrieves more context than most answers use -- roughly half the
  retrieved chunks go uncited in a typical answer.
- Claim extraction in `eval_citations.py` is line-based. Fragments ending in a
  colon and rendered formulas are filtered out; unusual formatting may not be.
