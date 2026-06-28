# Antigravity Agent Prompt — MOM Summarization Evaluation Pipeline

## Role
You are a senior ML engineer building an evaluation harness for a "Minutes of Meeting" (MOM)
generating agent. Your job is NOT to build the MOM generator itself — it already exists.
Your job is to objectively measure how good its summaries are, using the MeetingBank dataset
as ground truth.

## Objective
Build a reproducible Python evaluation pipeline that:
1. Acquires the MeetingBank dataset (transcript + reference summary pairs).
2. Runs our MOM-generation pipeline on the transcripts to produce candidate summaries.
3. Scores candidate summaries against reference summaries using multiple complementary metrics.
4. Produces a single evaluation report (table + JSON) with per-sample and aggregate scores.

---

## Step 1 — Dataset Acquisition
- Target dataset: **MeetingBank** (meeting transcripts + human-written summaries/minutes).
- Search Huggingface for it, download (Hugging Face `datasets` hub: `huangyz0918/meetingbank` or similar — verify the exact repo
  name before pulling) or the original paper's GitHub release.
- Validate the data: confirm each record has a `transcript` (or `segments`) field and a
  `summary`/`reference_summary` field. Log and drop malformed rows rather than failing silently.
- Save a cleaned, flattened version to `data/processed/meetingbank_clean.csv` with columns:
  `meeting_id, transcript, reference_summary`.

## Step 2 — Generate Candidate Summaries
- For each row in `meetingbank_clean.csv`, run the existing MOM pipeline (call its public
  function/API — do not reimplement it) to produce `candidate_summary`.
- Cache outputs to `data/processed/meetingbank_with_candidates.csv` so reruns of the eval
  step don't require regenerating summaries every time.
- Log generation latency per sample — useful operational signal even though it's not a
  quality metric.

## Step 3 — Evaluation Metrics
Use a mix of **lexical-overlap**, **semantic**, and **task-specific (MOM-aware)** metrics —
relying on ROUGE alone is not sufficient for meeting minutes because it rewards n-gram overlap
but says nothing about whether the right decisions/action items survived.

| Category | Metric | Why it's included | Library |
|---|---|---|---|
| Lexical overlap | ROUGE-1, ROUGE-2, ROUGE-L | Standard summarization benchmark metric; recall-oriented, captures content overlap | `rouge-score` or `evaluate` (HF) |
| Lexical overlap | BLEU | Precision-oriented sanity check; not primary, but useful as a cross-check against ROUGE | `nltk.translate.bleu_score` or `evaluate` |
| Lexical overlap | METEOR | Handles synonymy/stemming better than ROUGE/BLEU, more robust for paraphrased minutes | `evaluate` |
| Semantic similarity | BERTScore (P/R/F1) | Captures meaning-preserving paraphrases that ROUGE penalizes unfairly | `bert-score` |
| Semantic similarity | Cosine similarity of sentence embeddings (e.g. `all-MiniLM-L6-v2`) | Cheap, fast secondary semantic check, decoupled from BERTScore's internal model | `sentence-transformers` |
| Factual consistency | SummaC or FactCC-style entailment score | Detects hallucinated decisions/action items not present in the transcript — critical for MOM, where a false "decision" is worse than a missing one | `summac` |
| Compression | Compression ratio (`len(transcript)/len(summary)`) | Sanity metric — flags degenerate cases (e.g. summary is near copy-paste of transcript) | custom |
| Task-specific (MOM) | Action-item / decision recall & precision | Extract structured items (decision, owner, due date) from both reference and candidate via a lightweight NER/regex or LLM-extraction pass, then compute precision/recall/F1 on matched items | custom + optional LLM judge |
| Overall quality (optional) | LLM-as-judge (G-Eval style) scoring on coherence, relevance, conciseness, faithfulness (1–5 scale) | Cross-validates automatic metrics with a holistic quality signal; flag as "optional" since it's costly to run at full dataset scale | direct LLM call |

> Note: Report ROUGE/BLEU/METEOR/BERTScore for every sample. Run the factual-consistency and
> action-item metrics as a required core layer (not optional) — they're the metrics that
> actually matter for whether a MOM is *trustworthy*, not just fluent.

## Step 4 — Aggregation & Reporting
- Compute per-sample scores and store in `results/per_sample_scores.csv`.
- Compute mean/median/std for each metric across the dataset → `results/aggregate_scores.json`.
- Flag the bottom 5% of samples by BERTScore-F1 or factual-consistency score for manual review
  — these are likely cases of hallucination or missed key decisions.
- Produce a short markdown report (`results/eval_report.md`) summarizing:
  - Dataset size used, any rows dropped and why
  - Aggregate metric table
  - 3–5 worst-performing examples with transcript excerpt, reference, candidate, and why it
    scored low
  - Recommendations (e.g. "high ROUGE but low factual-consistency → model paraphrases well but
    invents action items")

## Constraints & Engineering Notes
- Python 3.10+, manage deps via `requirements.txt` (kaggle, datasets, evaluate, rouge-score,
  bert-score, sentence-transformers, summac, nltk, pandas).
- Make the pipeline resumable/cacheable — don't regenerate candidate summaries or re-download
  data on every run; check for existing processed files first.
- Parameterize dataset sample size (e.g. `--n_samples`) for quick smoke-testing before a full run,
  since MeetingBank is large and BERTScore/LLM-judge calls are expensive at scale.
- Handle missing Kaggle API credentials gracefully — print a clear setup instruction
  (`~/.kaggle/kaggle.json`) rather than a raw stack trace.
- All metric functions should be unit-testable in isolation (pass in two strings → return a score),
  decoupled from the data-loading/generation code.

## Deliverables
1. `eval_pipeline.py` — orchestrates steps 1–4 end-to-end.
2. `metrics.py` — all metric implementations, independently testable.
3. `results/eval_report.md` + `results/aggregate_scores.json` + `results/per_sample_scores.csv`.
4. `README.md` documenting how to run it and how to interpret each metric.
