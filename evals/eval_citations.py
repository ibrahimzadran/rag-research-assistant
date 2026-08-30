"""
eval_citations.py

Checks whether each cited claim in an answer is actually supported by the
source it cites -- the check you did by hand against the Ragas paper, automated.

This is the THIRD failure mode. It is distinct from retrieval (did we fetch the
right passage) and from faithfulness (did the answer stay inside the sources at
all). A model can be perfectly grounded in the retrieved text and still attach
the wrong bracket number to a claim, which is what makes a citation untrustworthy
even when the answer is correct.

Two numbers are reported:

  citation accuracy  Of the claims that carry a citation, how many are actually
                     supported by the source they point at. This is the headline
                     number: how much a bracket can be trusted.

  citation coverage  What fraction of substantive sentences carry any citation.
                     A high accuracy score means little if half the claims are
                     uncited -- the model could be smuggling in unsourced
                     assertions and scoring well on the ones it did cite.

Claims are extracted line by line, which suits the markdown the answer model
produces. Known limitation: when a bullet list sits under an introductory line
that carries the citation, the bullets read as uncited and land in the coverage
figure rather than the accuracy figure. That understates coverage rather than
inflating accuracy, so the headline number stays honest.

Usage:
    python evals/eval_citations.py --n 5

Cost: one Voyage call for all questions, then per question one answer plus one
judge call per cited claim.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

import sys

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv

# Import the REAL prompt builder from query_papers rather than copying it here.
# An eval that tests its own copy of the prompt measures a system you do not
# ship: improve the prompt in query_papers.py and the eval would keep scoring
# the old wording, reporting a number that describes nothing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from query_papers import build_prompt  # noqa: E402

load_dotenv()

EMBED_MODEL = "voyage-3.5"
CHAT_MODEL = "claude-sonnet-4-6"

CITATION = re.compile(r"\[(\d+)\]")
MIN_CLAIM_WORDS = 6  # below this a line is a heading or fragment, not a claim

# The judge is told to ignore what it already knows. Without that instruction an
# LLM judge will mark a claim SUPPORTED because the claim is true in the world,
# not because the source says it -- which would make this eval always pass and
# measure nothing.
JUDGE_PROMPT = """You are checking whether a claim is supported by a source text.

CLAIM:
{claim}

SOURCE TEXT:
{source}

Answer SUPPORTED only if the source text states the claim or directly implies it.
Answer NOT_SUPPORTED if the claim is absent from the source, contradicted by it,
or would require knowledge from outside the source -- even if you believe the
claim is true in general. Your own knowledge is irrelevant here; judge only
whether THIS text supports it.

Reply with exactly one word on the first line, SUPPORTED or NOT_SUPPORTED,
then one short sentence explaining why."""


def answer_question(claude, question, chunks):
    """Generate an answer using query_papers.py's own prompt, so the eval scores
    the same wording the real tool sends."""
    response = claude.messages.create(
        model=CHAT_MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def extract_claims(answer):
    """Split an answer into (text, [cited indices]) pairs, one per line.

    Returns cited claims and the count of substantive uncited lines. Lines that
    assert nothing -- headings, fragments introducing a list or formula, and
    rendered notation -- are dropped entirely rather than counted either way,
    since a judge cannot meaningfully verify them and their verdicts would
    describe the parser rather than the answer.
    """
    cited, uncited = [], 0
    for raw in answer.splitlines():
        if raw.strip().startswith("#"):
            continue
        line = raw.strip().lstrip("#-*> ").strip()
        if not line:
            continue

        refs = [int(n) for n in CITATION.findall(line)]
        # Judge the prose, not the bracket markers -- and test the filters
        # against the cleaned text, since a trailing "[3]" would otherwise hide
        # the colon that marks an introductory fragment.
        text = CITATION.sub("", line).strip()

        if len(text.split()) < MIN_CLAIM_WORDS:
            continue
        if text.rstrip().endswith(":"):
            continue
        # Rendered notation is not a natural-language claim. Counting letters
        # does not catch it, because "(N choose k)" spells out a word and reads
        # as prose; the giveaway is the density of math symbols instead.
        # Strip markdown emphasis first: ** and ` are formatting, not notation,
        # and counting them would discard real claims like "**EVOR** outperforms
        # **DocPrompting** by 18.6%".
        plain = text.replace("**", "").replace("`", "").replace("*", "")
        symbols = sum(ch in "()[]{}<>=+/\\^|" for ch in plain)
        if not plain or symbols / len(plain) > 0.08:
            continue

        if refs:
            cited.append((text, sorted(set(refs))))
        else:
            uncited += 1
    return cited, uncited


def judge(claude, claim, source_text):
    """Ask whether source_text supports claim. Returns (bool, reason)."""
    response = claude.messages.create(
        model=CHAT_MODEL, max_tokens=150,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            claim=claim, source=source_text)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    verdict = text.splitlines()[0].strip().upper()
    reason = " ".join(text.splitlines()[1:]).strip()
    return verdict.startswith("SUPPORTED"), reason


def main():
    parser = argparse.ArgumentParser(description="Score citation accuracy.")
    parser.add_argument("--n", type=int, default=5, help="How many questions to test")
    parser.add_argument("--testset", default="./evals/testset.jsonl")
    parser.add_argument("--db_dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_papers")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--results_dir", default="./evals/results")
    args = parser.parse_args()

    for key in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"Set {key} (in .env or the environment) first.")

    with open(args.testset, encoding="utf-8") as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()][:args.n]

    collection = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    vo = voyageai.Client()
    claude = anthropic.Anthropic()

    # All questions in one Voyage request -- avoids the per-minute request limit.
    vectors = vo.embed(questions, model=EMBED_MODEL, input_type="query").embeddings
    retrieved = collection.query(query_embeddings=vectors, n_results=args.top_k)

    records, supported_total, claims_total, uncited_total = [], 0, 0, 0

    for qi, question in enumerate(questions):
        chunks = [{"text": d, "meta": m} for d, m in
                  zip(retrieved["documents"][qi], retrieved["metadatas"][qi])]
        answer = answer_question(claude, question, chunks)
        cited, uncited = extract_claims(answer)
        uncited_total += uncited

        print(f"\nQ{qi + 1}: {question[:76]}")
        print(f"    {len(cited)} cited claim(s), {uncited} uncited line(s)")

        claim_records = []
        for claim, refs in cited:
            # A claim citing [1][3] is checked against both together, since
            # multiple sources may jointly support one statement.
            valid = [r for r in refs if 1 <= r <= len(chunks)]
            if not valid:
                # Cited a source number that does not exist -- an outright error.
                claims_total += 1
                claim_records.append({"claim": claim, "refs": refs,
                                      "supported": False,
                                      "reason": "cites a source number that was not retrieved"})
                print(f"      BAD REF {refs}  {claim[:62]}")
                continue

            source = "\n\n".join(f"[{r}] {chunks[r - 1]['text']}" for r in valid)
            ok, reason = judge(claude, claim, source)
            claims_total += 1
            supported_total += ok
            claim_records.append({"claim": claim, "refs": valid,
                                  "supported": ok, "reason": reason})
            print(f"      {'OK  ' if ok else 'FAIL'} {str(refs):8} {claim[:62]}")
            if not ok:
                print(f"            -> {reason[:88]}")

        records.append({"question": question, "answer": answer,
                        "uncited_lines": uncited, "claims": claim_records})

    accuracy = supported_total / claims_total if claims_total else 0.0
    total_lines = claims_total + uncited_total
    coverage = claims_total / total_lines if total_lines else 0.0

    print(f"\n{'':-<72}")
    print(f"citation accuracy .. {accuracy:.2f}  ({supported_total} of {claims_total} cited claims supported)")
    print(f"citation coverage .. {coverage:.2f}  ({claims_total} cited vs {uncited_total} uncited lines)")

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.results_dir, f"citations_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "top_k": args.top_k,
                   "citation_accuracy": accuracy, "citation_coverage": coverage,
                   "claims_checked": claims_total, "uncited_lines": uncited_total,
                   "questions": records}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
