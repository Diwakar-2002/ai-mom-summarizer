# Revision Request — MOM Evaluation Pipeline Implementation Plan

Your previous implementation plan is approved in structure, but needs the following fixes
**before** you start writing code. Please revise the plan and re-share it for review.

---

## 1. Fix the duplicated step
"Step 4: Aggregate Results" is listed three times in the Proposed Changes section. Clean this
up — it should appear once.

## 2. Separate "smoke test" sample size from "real evaluation" sample size
`--n_samples` defaulting to 5 is fine for a smoke test, but mean/median/std over 5 samples is
not a trustworthy evaluation result — a single outlier transcript will swing the aggregates.

- Keep `--n_samples 5` as the documented smoke-test value.
- Set the actual default for a real evaluation run to something in the 100–200 range.
- Document both modes explicitly in README.md so it's clear a 5-sample run is for pipeline
  validation only, not a reportable benchmark result.

## 3. Add a fallback for SummaC
SummaC has brittle, easily outdated dependency pinning (specific torch/transformers versions)
and is a realistic install/runtime failure point.

- Wrap SummaC usage so that if it fails to import or run, `compute_factual_consistency` falls
  back to a plain NLI-entailment check using a `transformers` pipeline (e.g. a DeBERTa-MNLI
  zero-shot/entailment model) treating the transcript as premise and summary as hypothesis.
- Log clearly which method (SummaC vs NLI fallback) was used for a given run, since the two
  aren't numerically comparable — don't silently mix them within one aggregate report.

## 4. Commit to a concrete action-item extraction method (not regex/heuristic)
"Custom NER/regex or heuristic extraction" is too vague and won't survive paraphrasing (e.g.
"ask John to send the deck by Friday" vs "John: send deck, due Fri" — regex won't match these
as the same action item).

Replace this with:
- Use structured LLM extraction to pull `{action, owner, deadline}` triples from both the
  reference summary and the candidate summary (a small JSON-output prompt call — can reuse the
  same local phi3/Gemini setup already in use for generation).
- Match extracted triples between reference and candidate using embedding similarity (e.g.
  `sentence-transformers` cosine similarity above a threshold) rather than exact string match.
- Compute precision/recall/F1 over matched vs unmatched triples.
- Document the matching threshold used and make it configurable.

## 5. Fix the caching logic to key off sample size / row IDs
Currently the plan checks only "does the cache file exist," which breaks if `--n_samples` is
increased on a later run (it would silently reuse the smaller cached file rather than
extending it).

- Cache should be keyed by row/meeting ID, not just file existence.
- If a run requests more samples than are currently cached, generate only the missing rows and
  append to the existing cache rather than regenerating everything or silently truncating.

## 6. Add run versioning for results
Every run currently overwrites `results/eval_report.md`, `aggregate_scores.json`, and
`per_sample_scores.csv`. Since the MOM prompts are actively being iterated on, we need to
compare runs over time.

- Write outputs to a timestamped or tag-based subfolder, e.g. `results/run_2026-06-26_1140/`,
  and optionally maintain a `results/latest/` symlink or copy for convenience.
- Add a small comparison helper (can be a follow-up, but mention it in the README as planned)
  that diffs aggregate scores between two run folders.

## 7. Add a baseline/sanity anchor metric
Add at least one trivial baseline alongside the real candidate summaries so absolute metric
values have something to compare against — e.g.:
- "Lead-N" baseline: first N sentences of the transcript used as a naive summary.
- Reference-vs-reference: compute every metric of the reference summary against itself (should
  produce near-ceiling scores) as a check that the metric functions themselves are working
  correctly.

Report baseline scores alongside the candidate scores in `eval_report.md` for context.

## 8. Confirm model choice handling
Keep local phi3 as the default candidate-generation model, since the goal is evaluating the
actual production MOM pipeline, not a hypothetical stronger one. Add an optional `--model
gemini` flag (using `GEMINI_API_KEY` from `.env` if present) purely for side-by-side
comparison runs — not as the default evaluation target. Clearly label which model was used to
generate candidates in every output report.

---

Please update the implementation plan to reflect all of the above, then re-share it for
approval before writing any code.
