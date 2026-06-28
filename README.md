# Minutes of Meeting (MOM) Summarization Evaluation Pipeline

This repository contains a reproducible, automated evaluation pipeline to measure the quality of our Minutes of Meeting (MOM) summarization model against the human-annotated **MeetingBank** dataset.

---

## 📋 Features

- **Dataset Acquisition**: Automatically downloads, validates, and cleans the MeetingBank dataset, caching it to prevent redundant downloads.
- **Incremental Cacheable Summaries**: Runs transcripts through the summarization script and caches candidate summaries by `meeting_id` and `model` to allow extending runs without reprocessing.
- **Trivial Baselines**: Anchors evaluation scores using:
  - **Lead-N**: First $N$ sentences of the transcript (naive summary baseline).
  - **Reference-vs-Reference**: The ground-truth reference summary evaluated against itself (ceiling sanity baseline).
- **Comprehensive Metric Suite**:
  - *Lexical Overlap*: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR.
  - *Semantic Similarity*: BERTScore (P/R/F1) and Sentence Embedding Cosine Similarity (using `all-MiniLM-L6-v2`).
  - *Factual Consistency*: SummaC evaluator with a robust zero-shot NLI entailment fallback (`facebook/bart-large-mnli`).
  - *Compression Ratio*: Measures details reduction ratio.
  - *Action Item F1*: Extracts `{action, owner, deadline}` triples using LLM parsing, matches them semantically using embedding cosine similarity, and computes precision, recall, and F1.
- **Run Versioning**: Saves all evaluation results to timestamped folders under `results/run_YYYY-MM-DD_HHMM/` with a `results/latest/` symlink/copy for convenient scripting.
- **Automatic Diagnostics**: Identifies and logs the bottom 5% of candidate summaries for manual review.

---

## 🛠️ Setup

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Ollama Server (for Local Models)**:
   Make sure Ollama is running and has the default model downloaded:
   ```bash
   # In a separate terminal
   ./bin/ollama serve
   
   # Pull the required model
   ollama pull phi3
   ```
3. **Configure API Keys (Optional)**:
   If running comparison runs with Gemini, add your `GEMINI_API_KEY` to the `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 🚀 Running the Pipeline

### Mode 1: Smoke-Test Mode (Pipeline Validation)
To verify that everything is working properly without consuming extensive compute/API tokens:
```bash
python3 eval_pipeline.py --n_samples 5
```
> [!NOTE]  
> A 5-sample run is strictly for pipeline validation and code sanity check. It is **not** a statistically significant evaluation result.

### Mode 2: Full Evaluation Mode (Benchmark)
To generate the real report and obtain statistically sound aggregate metrics:
```bash
python3 eval_pipeline.py --n_samples 100
```
*(You can scale this up to `--n_samples 200` as needed).*

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--n_samples` | Number of samples to download and evaluate. | `100` |
| `--model` | Model to generate candidate summaries (`phi3` or `gemini`). | `phi3` |
| `--threshold` | Cosine similarity threshold for action item embedding matching. | `0.7` |
| `--n_lead` | Number of sentences to take for the Lead-N baseline summary. | `3` |

---

## 📊 Result Artifacts

Every run creates a timestamped subdirectory under `results/` containing:
- **`eval_report.md`**: A detailed report comparing candidate summaries, Lead-N, and Reference-vs-Reference baselines, complete with flagged low-performing outlier examples.
- **`aggregate_scores.json`**: JSON file containing the mean, median, and standard deviation of all metrics.
- **`per_sample_scores.csv`**: A spreadsheet listing all metrics for every individual sample.

A copy of the latest run is maintained in `results/latest/` for convenience.

---

## 📐 Interpreting the Metrics

### 1. Lexical Overlap (ROUGE / BLEU / METEOR)
- **ROUGE-L & METEOR**: Higher scores mean the model captures the key vocabulary and structures of the meeting minutes. METEOR handles stemming and synonyms better, which rewards good paraphrasing.
- **BLEU**: A precision-focused overlap metric; useful to cross-verify that the summary isn't generating junk n-grams.

### 2. Semantic Similarity (BERTScore / Sentence Embedding Cosine)
- Measures whether the *meaning* is preserved, even if the model uses entirely different phrasing. A high BERTScore with a low ROUGE score means the model paraphrases well.

### 3. Factual Consistency
- **Factual Consistency**: A score between $0.0$ and $1.0$. SummaC/NLI models check whether statements in the summary are logically entailed by the transcript. A low score indicates potential hallucinations (fabricating decisions or numbers).
- **Factual Method**: Indicates if `summac` was used or if it failed over to the CPU-friendly `nli_fallback` (BART-MNLI). Note that SummaC and NLI scores are not numerically comparable, so comparison runs must check that the same method was used.

### 4. Action-Item F1
- Measures how accurately the model extracts responsibilities. We pull `{action, owner, deadline}` triples and match the actions using semantic embedding cosine similarity (above `--threshold`, default $0.7$).
  - **High Precision, Low Recall**: The model is conservative—it only extracts action items it is highly confident in, but misses several.
  - **Low Precision, High Recall**: The model is over-aggressive—it extracts many items, but many are redundant, vague, or not actually action items.

---

## 🔄 Planned Features: Comparison Helper

We plan to release a diff helper `diff_runs.py` to allow automated comparison of aggregate scores between two runs:
```bash
# Compare local phi3 run against gemini run
python3 diff_runs.py results/run_2026-06-26_1140 results/run_2026-06-26_1215
```
This will print a markdown table highlighting the absolute delta ($\Delta$) for each metric, helping you instantly see if a prompt change or fine-tuned model yields a statistically significant quality improvement.
