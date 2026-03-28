# FahMai RAG Challenge

This repository contains our working solution for the `Super AI Engineer S6 - FahMai RAG Challenge (level 1)`.

Current canonical submission:
- [`submission.csv`](submission.csv)

Public-best reference snapshot:
- [`artifacts/analysis/submission_fix_q49_20260328.csv`](artifacts/analysis/submission_fix_q49_20260328.csv)

These two files are currently identical and represent the `1.00` public-score baseline we want to preserve.

## Problem

The challenge is multiple-choice QA over a fixed FahMai knowledge base:
- product pages
- store info
- warranty / return / shipping / points policies

Each question has choices `1-10`:
- `1-8` are concrete answers
- `9` means the answer is not available in the FahMai knowledge base
- `10` means the question is outside the scope of the FahMai database

## Our Path

We did not get to `1.00` with a single raw model run.

The path was:
1. Start from a standard RAG baseline.
2. Improve retrieval with hybrid search, reranking, markdown chunking, and source-aware retrieval.
3. Observe that model-only runs still failed on exact facts, compare questions, policy clauses, and calculation-style questions.
4. Add stricter retrieval shaping:
   - faceted filtering
   - exact-fact aware context handling
   - better routing for in-box / availability / policy / compare queries
5. Change the answering flow from `rule first` to `LLM first, review second`.
6. Add an evidence-aligned review layer that:
   - extracts structured facts from KB evidence
   - sends those facts back to the LLM for a retry
   - only falls back to a deterministic evidence rule when the LLM still conflicts with the evidence

That final shape is the version we keep in the main pipeline.

## Score Progression

The public-score path that led to the final solution:

| Submission file | Public score |
|---|---:|
| `preview_submission.csv` | `0.15` |
| `full_submission_heuristic.csv` | `0.41` |
| `submission.csv` | `0.48` |
| `submission_graphrag_prompt_v2.csv` | `0.56` |
| `submission_stable_hybrid.csv` | `0.63` |
| `submission_graphrag.csv` | `0.65` |
| `context_window1_smoke.csv` | `0.10` |
| `submission_context_window1.csv` | `0.68` |
| `submission.csv` | `0.80` |
| `submission.csv` | `0.75` |
| `submission.csv` | `0.76` |
| `super_final.csv` | `0.90` |
| `god.csv` | `0.95` |
| `super_final.csv` | `0.98` |
| `submission_fix_q49_20260328.csv` | `1.00` |

The important takeaway is that the final jump did not come from one isolated model change. It came from tightening retrieval, then adding an evidence-aligned review layer that corrected exact-fact, compare, policy, and calculation failures while still keeping the LLM in the loop.

## Final Solution Shape

The current system in [`fahmai_rag_jina.py`](fahmai_rag_jina.py) is:

1. Load questions and KB documents.
2. Chunk markdown documents with overlap.
3. Retrieve candidates using hybrid retrieval.
4. Apply source-aware retrieval and faceted filtering.
5. Rerank candidates.
6. Compress context selectively for prompt quality.
7. Ask the LLM first.
8. Run evidence review:
   - structured fact extraction from matched evidence
   - retry prompt to the LLM with structured facts
   - final fallback to evidence rule only if needed

This keeps the system aligned with a real RAG workflow rather than turning it into a pure answer-key script.

## Main Files

- [`fahmai_rag_jina.py`](fahmai_rag_jina.py)
  Main pipeline.

- [`fahmai_rag_jina.ipynb`](fahmai_rag_jina.ipynb)
  Notebook version synced from the Python pipeline.

- [`submission.csv`](submission.csv)
  Canonical `1.00` submission. Do not overwrite casually.

- [`data/questions.csv`](data/questions.csv)
  Challenge questions.

- [`data/knowledge_base`](data/knowledge_base)
  Full FahMai KB used for retrieval.

- [`artifacts/analysis`](artifacts/analysis)
  All experiments, candidate CSVs, debug logs, and comparison snapshots.

- [`scripts`](scripts)
  Utility scripts used during analysis and pseudo-label generation.

- [`archive/submissions`](archive/submissions)
  Old submission variants and scratch outputs kept for reference.

## Recommended Run

The current recommended candidate run is:

```bash
export THAILLM_API_KEY="..."

uv run python "/Users/chinnphats/Desktop/cedt/Super AI/Competetion/Hack3/fahmai_rag_jina.py" \
  --llm-model openthaigpt \
  --embedding-model jinaai/jina-embeddings-v5-text-small \
  --retriever hybrid \
  --reranker \
  --source-aware-retrieval \
  --faceted-filtering \
  --context-compression \
  --deterministic-solvers \
  --no-query-planning \
  --no-choice-verifier \
  --chunking-strategy markdown \
  --chunk-size 512 \
  --chunk-overlap 128 \
  --top-k 5 \
  --fetch-k 12 \
  --hint-chunks-per-source 1 \
  --max-per-source 2 \
  --candidate-max-per-source 3 \
  --request-timeout 45 \
  --max-retries 2 \
  --debug-log "/Users/chinnphats/Desktop/cedt/Super AI/Competetion/Hack3/artifacts/analysis/candidate_debug.jsonl" \
  --output "/Users/chinnphats/Desktop/cedt/Super AI/Competetion/Hack3/artifacts/analysis/candidate.csv"
```

Important:
- write new candidates to `artifacts/analysis/`
- do not overwrite [`submission.csv`](submission.csv) until the candidate is checked

## Compare A Candidate Against The Canonical Submission

```bash
python3 - <<'PY'
import csv
from pathlib import Path

base = Path("/Users/chinnphats/Desktop/cedt/Super AI/Competetion/Hack3/submission.csv")
cand = Path("/Users/chinnphats/Desktop/cedt/Super AI/Competetion/Hack3/artifacts/analysis/candidate.csv")

def load(path):
    with path.open() as f:
        return {int(r["id"]): int(r["answer"]) for r in csv.DictReader(f)}

b = load(base)
c = load(cand)
diffs = [(i, c.get(i), b.get(i)) for i in sorted(b) if c.get(i) != b.get(i)]
print("match", 100 - len(diffs), "/ 100")
print("diffs", diffs)
PY
```

## Notes On Models

What we learned during development:
- `Pathumma` sometimes looked stronger on subsets, but endpoint stability was poor.
- `OpenThaiGPT` was more practical for repeated iteration.
- pure model-only RAG did not reproduce the `1.00` submission reliably
- retrieval quality mattered, but final answer selection mattered more

That is why the final system is not just `retrieve -> prompt -> answer`. It is `retrieve -> prompt -> evidence review -> retry -> fallback`.

## Notebook Workflow

If you prefer notebooks:
- open [`fahmai_rag_jina.ipynb`](fahmai_rag_jina.ipynb)
- the notebook is synced from the script
- it contains the full pipeline code plus a run cell template

There is also a lightweight runner notebook here:
- [`output/jupyter-notebook/fahmai-submission-runner.ipynb`](output/jupyter-notebook/fahmai-submission-runner.ipynb)

## Project Organization

Root should stay clean and hold only the important entry points:
- pipeline
- notebook
- canonical submission
- configs

Everything experimental should go under:
- [`artifacts/analysis`](artifacts/analysis)
- [`archive/submissions`](archive/submissions)

## Status

Current state:
- canonical public submission is locked at `1.00`
- repo is organized
- main Python pipeline and notebook are synced
- future work should be done through candidate files under `artifacts/analysis/`
