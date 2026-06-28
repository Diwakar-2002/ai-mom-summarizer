import os
import re
import sys
import json
import time
import argparse
import logging
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Import code from summarize and metrics
from summarize import generate_summary
from metrics import (
    compute_rouge,
    compute_bleu,
    compute_meteor,
    compute_bertscore,
    compute_semantic_cosine,
    compute_factual_consistency,
    compute_compression_ratio,
    compute_action_item_metrics
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_lead_n_baseline(transcript: str, n: int = 3) -> str:
    """Generates Lead-N baseline: first N sentences of the transcript."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
    sentences = nltk.sent_tokenize(transcript.strip())
    return " ".join(sentences[:n])

def call_llm_json(prompt: str, model_name: str) -> list:
    """
    Queries the LLM (Ollama or Gemini) with a structured prompt
    and parses/returns a JSON list.
    """
    import requests
    if model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            logging.warning("GEMINI_API_KEY not found. Action item LLM extraction skipped or defaulted to empty list.")
            return []
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logging.error(f"Gemini API request failed for action extraction: {e}")
            return []
    else:
        # Local Ollama
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,
                "top_p": 0.9,
                "num_predict": 500
            }
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            text = response.json().get("response", "")
        except Exception as e:
            logging.error(f"Ollama API request failed for action extraction: {e}. Make sure Ollama is running.")
            return []

    # Clean up and parse JSON
    try:
        clean_text = text.strip()
        match = re.search(r'(\[.*\]|\{.*\})', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            
        parsed = json.loads(clean_text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            for key in ["action_items", "actions", "triples"]:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]
        return []
    except Exception as e:
        logging.warning(f"Failed to parse LLM action item extraction: {e}. Raw response: {text}")
        try:
            dict_pattern = r'\{[^{}]*\}'
            dicts = re.findall(dict_pattern, text)
            parsed_list = []
            for d in dicts:
                try:
                    parsed_list.append(json.loads(d))
                except:
                    pass
            return parsed_list
        except:
            return []

def extract_action_items_from_summary(summary_text: str, model_name: str) -> list:
    """Extracts action items as a list of triples {action, owner, deadline} from summary."""
    if not summary_text.strip():
        return []
        
    prompt = f"""You are a precise data extraction assistant.
Read the following meeting summary and extract all action items and decisions.
For each action item or decision, extract:
1. "action": What needs to be done.
2. "owner": The specific person, role, or team assigned to it. If not mentioned, set this to "None".
3. "deadline": The deadline or date. If not mentioned, set this to "None".

OUTPUT FORMAT:
Your output must be a valid JSON list of objects. Do not write any markdown formatting, preambles, explanations, or conclusions. Just return the JSON list.

Example Output:
[
  {{"action": "Submit final budget report", "owner": "CFO", "deadline": "June 30"}},
  {{"action": "Schedule follow up meeting", "owner": "None", "deadline": "None"}}
]

Summary:
{summary_text}
"""
    return call_llm_json(prompt, model_name)

def acquire_and_clean_data(n_samples: int) -> pd.DataFrame:
    """Acquires the MeetingBank dataset, cleans/flattens it, and saves to data/processed/meetingbank_clean.csv"""
    clean_csv_path = "data/processed/meetingbank_clean.csv"
    os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)
    
    if os.path.exists(clean_csv_path):
        logging.info(f"Loading cleaned dataset from: {clean_csv_path}")
        df = pd.read_csv(clean_csv_path)
    else:
        logging.info("Downloading MeetingBank dataset from Hugging Face hub...")
        try:
            dataset_dict = load_dataset("huuuyeah/meetingbank")
            split_name = 'test'
            if split_name not in dataset_dict:
                available = list(dataset_dict.keys())
                logging.warning(f"'test' split not found. Using '{available[0]}'.")
                split_name = available[0]
            dataset = dataset_dict[split_name]
        except Exception as e:
            logging.error(f"HuggingFace dataset download failed: {e}")
            raise
            
        logging.info(f"Loaded {len(dataset)} records from HF split '{split_name}'. Cleaning...")
        records = []
        dropped_count = 0
        
        for idx, row in enumerate(dataset):
            transcript = None
            for key in ['transcript', 'text', 'document', 'source']:
                if key in row and row[key]:
                    transcript = str(row[key]).strip()
                    break
                    
            summary = None
            for key in ['summary', 'reference_summary', 'target', 'reference']:
                if key in row and row[key]:
                    summary = str(row[key]).strip()
                    break
                    
            meeting_id = None
            for key in ['meeting_id', 'id', 'meetingId']:
                if key in row and row[key]:
                    meeting_id = str(row[key]).strip()
                    break
            if not meeting_id:
                meeting_id = f"mb_{idx}"
                
            if not transcript or not summary:
                dropped_count += 1
                continue
                
            records.append({
                "meeting_id": meeting_id,
                "transcript": transcript,
                "reference_summary": summary
            })
            
        logging.info(f"Cleaned {len(records)} records. Dropped {dropped_count} empty records.")
        df = pd.DataFrame(records)
        df.to_csv(clean_csv_path, index=False)
        logging.info(f"Saved clean dataset to: {clean_csv_path}")
        
    return df

def generate_and_cache_candidates(df_clean: pd.DataFrame, n_samples: int, model_name: str) -> pd.DataFrame:
    """
    Generates candidate summaries for n_samples using summarize.py.
    Caches outputs incrementally in data/processed/meetingbank_with_candidates.csv keyed by meeting_id and model.
    """
    cache_path = "data/processed/meetingbank_with_candidates.csv"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["meeting_id", "model", "candidate_summary", "latency"])
        
    cache_dict = {}
    for _, row in cache_df.iterrows():
        m_id = str(row['meeting_id'])
        m_model = str(row['model'])
        cache_dict[(m_id, m_model)] = (str(row['candidate_summary']), float(row['latency']))
        
    subset_df = df_clean.head(n_samples).copy()
    results = []
    new_rows = []
    
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f"Acquiring Candidates ({model_name})"):
        m_id = str(row['meeting_id'])
        key = (m_id, model_name)
        
        if key in cache_dict:
            cand_summary, latency = cache_dict[key]
        else:
            transcript = str(row['transcript'])
            logging.info(f"Generating summary for meeting {m_id} using {model_name}...")
            start_time = time.time()
            try:
                cand_summary = generate_summary(transcript, meeting_type="general", model_name=model_name)
                latency = time.time() - start_time
            except Exception as e:
                logging.error(f"Summary generation failed for {m_id}: {e}")
                cand_summary = ""
                latency = 0.0
                
            new_rows.append({
                "meeting_id": m_id,
                "model": model_name,
                "candidate_summary": cand_summary,
                "latency": latency
            })
            cache_dict[key] = (cand_summary, latency)
            
        results.append({
            "meeting_id": m_id,
            "transcript": row['transcript'],
            "reference_summary": row['reference_summary'],
            "candidate_summary": cand_summary,
            "latency": latency
        })
        
    if new_rows:
        new_cache_df = pd.DataFrame(new_rows)
        if os.path.exists(cache_path):
            new_cache_df.to_csv(cache_path, mode='a', header=False, index=False)
        else:
            new_cache_df.to_csv(cache_path, index=False)
        logging.info(f"Appended {len(new_rows)} items to candidate cache: {cache_path}")
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MOM Summarization Pipeline")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--model", type=str, default="phi3", choices=["phi3", "gemini"], help="Candidate generator model (default: phi3)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Action-item cosine similarity matching threshold (default: 0.7)")
    parser.add_argument("--n_lead", type=int, default=3, help="Sentence length for Lead-N baseline summary (default: 3)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    try:
        df_clean = acquire_and_clean_data(args.n_samples)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    if len(df_clean) < args.n_samples:
        logging.warning(f"Requested {args.n_samples} samples, but only {len(df_clean)} are available. Evaluating all {len(df_clean)} samples.")
        args.n_samples = len(df_clean)
        
    df_eval = generate_and_cache_candidates(df_clean, args.n_samples, args.model)
    
    if args.model == "phi3":
        import requests
        try:
            requests.get("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            print("\n" + "="*80)
            print("WARNING: Could not connect to local Ollama server at http://localhost:11434.")
            print("If you have uncached samples, generation will fail.")
            print("Please run 'ollama serve' in another terminal to start it.")
            print("="*80 + "\n")

    logging.info(f"Beginning evaluation on {args.n_samples} samples...")
    
    rows = []
    factual_consistency_methods = set()
    
    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Evaluating Samples"):
        meeting_id = row['meeting_id']
        transcript = row['transcript']
        ref = row['reference_summary']
        cand = row['candidate_summary']
        latency = row['latency']
        
        lead_n = get_lead_n_baseline(transcript, args.n_lead)
        
        logging.info(f"[{meeting_id}] Extracting action item triples...")
        ref_triples = extract_action_items_from_summary(ref, args.model)
        cand_triples = extract_action_items_from_summary(cand, args.model)
        lead_n_triples = extract_action_items_from_summary(lead_n, args.model)
        
        targets = {
            "candidate": (cand, cand_triples, latency),
            "lead_n": (lead_n, lead_n_triples, 0.0),
            "ref_vs_ref": (ref, ref_triples, 0.0)
        }
        
        scores_sample = {"meeting_id": meeting_id}
        
        for prefix, (text, triples, lat) in targets.items():
            rouge_scores = compute_rouge(ref, text)
            bleu = compute_bleu(ref, text)
            meteor = compute_meteor(ref, text)
            bert_scores = compute_bertscore(ref, text)
            sem_cos = compute_semantic_cosine(ref, text)
            comp = compute_compression_ratio(transcript, text)
            if prefix == "ref_vs_ref":
                fact_score, fact_method = 1.0, "ceiling"
            else:
                res_fact = compute_factual_consistency(transcript, text)
                fact_score = res_fact["factual_consistency"]
                fact_method = res_fact["method"]
                factual_consistency_methods.add(fact_method)
                
            ai_scores = compute_action_item_metrics(ref_triples, triples, threshold=args.threshold)
            
            scores_sample[f"{prefix}_rouge1"] = rouge_scores["rouge1"]
            scores_sample[f"{prefix}_rouge2"] = rouge_scores["rouge2"]
            scores_sample[f"{prefix}_rougeL"] = rouge_scores["rougeL"]
            scores_sample[f"{prefix}_bleu"] = bleu
            scores_sample[f"{prefix}_meteor"] = meteor
            scores_sample[f"{prefix}_bertscore_p"] = bert_scores["bertscore_p"]
            scores_sample[f"{prefix}_bertscore_r"] = bert_scores["bertscore_r"]
            scores_sample[f"{prefix}_bertscore_f1"] = bert_scores["bertscore_f1"]
            scores_sample[f"{prefix}_semantic_cosine"] = sem_cos
            scores_sample[f"{prefix}_factual_consistency"] = fact_score
            scores_sample[f"{prefix}_factual_method"] = fact_method
            scores_sample[f"{prefix}_compression_ratio"] = comp
            scores_sample[f"{prefix}_action_item_precision"] = ai_scores["action_item_precision"]
            scores_sample[f"{prefix}_action_item_recall"] = ai_scores["action_item_recall"]
            scores_sample[f"{prefix}_action_item_f1"] = ai_scores["action_item_f1"]
            
        scores_sample["candidate_latency"] = latency
        rows.append(scores_sample)
        
    df_scores = pd.DataFrame(rows)
    
    aggregates = {}
    metrics_to_agg = [
        "rouge1", "rouge2", "rougeL", "bleu", "meteor", 
        "bertscore_p", "bertscore_r", "bertscore_f1", 
        "semantic_cosine", "factual_consistency", "compression_ratio",
        "action_item_precision", "action_item_recall", "action_item_f1"
    ]
    
    for prefix in ["candidate", "lead_n", "ref_vs_ref"]:
        aggregates[prefix] = {}
        for m in metrics_to_agg:
            col = f"{prefix}_{m}"
            vals = df_scores[col].dropna()
            aggregates[prefix][m] = {
                "mean": float(vals.mean()) if len(vals) > 0 else 0.0,
                "median": float(vals.median()) if len(vals) > 0 else 0.0,
                "std": float(vals.std()) if len(vals) > 1 else 0.0
            }
            
    cand_latencies = df_scores["candidate_latency"].dropna()
    aggregates["candidate_latency"] = {
        "mean": float(cand_latencies.mean()) if len(cand_latencies) > 0 else 0.0,
        "median": float(cand_latencies.median()) if len(cand_latencies) > 0 else 0.0,
        "std": float(cand_latencies.std()) if len(cand_latencies) > 1 else 0.0
    }
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = f"results/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    per_sample_path = os.path.join(run_dir, "per_sample_scores.csv")
    agg_path = os.path.join(run_dir, "aggregate_scores.json")
    report_path = os.path.join(run_dir, "eval_report.md")
    
    df_scores.to_csv(per_sample_path, index=False)
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(aggregates, f, indent=2)
        
    bottom_5_percent = int(max(1, np.ceil(args.n_samples * 0.05)))
    df_sorted_bert = df_scores.sort_values(by="candidate_bertscore_f1", ascending=True)
    flagged_samples = df_sorted_bert.head(bottom_5_percent)
    
    factual_consistency_method_str = ", ".join(factual_consistency_methods) if factual_consistency_methods else "none"
    report_content = f"""# Minutes of Meeting (MOM) Evaluation Report
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Candidate Generator Model**: `{args.model}`
**Dataset**: MeetingBank (Evaluation Sample Size: {args.n_samples})
**Factual Consistency Evaluator Method**: `{factual_consistency_method_str}`
**Action Item Sim Threshold**: `{args.threshold}`

---

## Aggregate Scores

| Metric | Candidate Mean (Std) | Lead-N Mean (Std) | Ref-vs-Ref Mean (Std) |
|---|---|---|---|
| **ROUGE-1** | {aggregates['candidate']['rouge1']['mean']:.4f} ({aggregates['candidate']['rouge1']['std']:.4f}) | {aggregates['lead_n']['rouge1']['mean']:.4f} ({aggregates['lead_n']['rouge1']['std']:.4f}) | {aggregates['ref_vs_ref']['rouge1']['mean']:.4f} ({aggregates['ref_vs_ref']['rouge1']['std']:.4f}) |
| **ROUGE-2** | {aggregates['candidate']['rouge2']['mean']:.4f} ({aggregates['candidate']['rouge2']['std']:.4f}) | {aggregates['lead_n']['rouge2']['mean']:.4f} ({aggregates['lead_n']['rouge2']['std']:.4f}) | {aggregates['ref_vs_ref']['rouge2']['mean']:.4f} ({aggregates['ref_vs_ref']['rouge2']['std']:.4f}) |
| **ROUGE-L** | {aggregates['candidate']['rougeL']['mean']:.4f} ({aggregates['candidate']['rougeL']['std']:.4f}) | {aggregates['lead_n']['rougeL']['mean']:.4f} ({aggregates['lead_n']['rougeL']['std']:.4f}) | {aggregates['ref_vs_ref']['rougeL']['mean']:.4f} ({aggregates['ref_vs_ref']['rougeL']['std']:.4f}) |
| **BLEU** | {aggregates['candidate']['bleu']['mean']:.4f} ({aggregates['candidate']['bleu']['std']:.4f}) | {aggregates['lead_n']['bleu']['mean']:.4f} ({aggregates['lead_n']['bleu']['std']:.4f}) | {aggregates['ref_vs_ref']['bleu']['mean']:.4f} ({aggregates['ref_vs_ref']['bleu']['std']:.4f}) |
| **METEOR** | {aggregates['candidate']['meteor']['mean']:.4f} ({aggregates['candidate']['meteor']['std']:.4f}) | {aggregates['lead_n']['meteor']['mean']:.4f} ({aggregates['lead_n']['meteor']['std']:.4f}) | {aggregates['ref_vs_ref']['meteor']['mean']:.4f} ({aggregates['ref_vs_ref']['meteor']['std']:.4f}) |
| **BERTScore F1** | {aggregates['candidate']['bertscore_f1']['mean']:.4f} ({aggregates['candidate']['bertscore_f1']['std']:.4f}) | {aggregates['lead_n']['bertscore_f1']['mean']:.4f} ({aggregates['lead_n']['bertscore_f1']['std']:.4f}) | {aggregates['ref_vs_ref']['bertscore_f1']['mean']:.4f} ({aggregates['ref_vs_ref']['bertscore_f1']['std']:.4f}) |
| **Embedding Cosine** | {aggregates['candidate']['semantic_cosine']['mean']:.4f} ({aggregates['candidate']['semantic_cosine']['std']:.4f}) | {aggregates['lead_n']['semantic_cosine']['mean']:.4f} ({aggregates['lead_n']['semantic_cosine']['std']:.4f}) | {aggregates['ref_vs_ref']['semantic_cosine']['mean']:.4f} ({aggregates['ref_vs_ref']['semantic_cosine']['std']:.4f}) |
| **Factual Consistency** | {aggregates['candidate']['factual_consistency']['mean']:.4f} ({aggregates['candidate']['factual_consistency']['std']:.4f}) | {aggregates['lead_n']['factual_consistency']['mean']:.4f} ({aggregates['lead_n']['factual_consistency']['std']:.4f}) | {aggregates['ref_vs_ref']['factual_consistency']['mean']:.4f} ({aggregates['ref_vs_ref']['factual_consistency']['std']:.4f}) |
| **Action-Item F1** | {aggregates['candidate']['action_item_f1']['mean']:.4f} ({aggregates['candidate']['action_item_f1']['std']:.4f}) | {aggregates['lead_n']['action_item_f1']['mean']:.4f} ({aggregates['lead_n']['action_item_f1']['std']:.4f}) | {aggregates['ref_vs_ref']['action_item_f1']['mean']:.4f} ({aggregates['ref_vs_ref']['action_item_f1']['std']:.4f}) |
| **Action-Item Prec** | {aggregates['candidate']['action_item_precision']['mean']:.4f} ({aggregates['candidate']['action_item_precision']['std']:.4f}) | {aggregates['lead_n']['action_item_precision']['mean']:.4f} ({aggregates['lead_n']['action_item_precision']['std']:.4f}) | {aggregates['ref_vs_ref']['action_item_precision']['mean']:.4f} ({aggregates['ref_vs_ref']['action_item_precision']['std']:.4f}) |
| **Action-Item Rec** | {aggregates['candidate']['action_item_recall']['mean']:.4f} ({aggregates['candidate']['action_item_recall']['std']:.4f}) | {aggregates['lead_n']['action_item_recall']['mean']:.4f} ({aggregates['lead_n']['action_item_recall']['std']:.4f}) | {aggregates['ref_vs_ref']['action_item_recall']['mean']:.4f} ({aggregates['ref_vs_ref']['action_item_recall']['std']:.4f}) |
| **Compression Ratio** | {aggregates['candidate']['compression_ratio']['mean']:.4f} ({aggregates['candidate']['compression_ratio']['std']:.4f}) | {aggregates['lead_n']['compression_ratio']['mean']:.4f} ({aggregates['lead_n']['compression_ratio']['std']:.4f}) | {aggregates['ref_vs_ref']['compression_ratio']['mean']:.4f} ({aggregates['ref_vs_ref']['compression_ratio']['std']:.4f}) |

**Operational Performance**:
*   Average generation latency: `{aggregates['candidate_latency']['mean']:.2f} seconds` (std: `{aggregates['candidate_latency']['std']:.2f}s`)

---

## Flagged Samples (Bottom 5% by BERTScore F1)

These samples scored lowest and are prime candidates for manual review to detect hallucinations or severe missing content.

"""
    for _, flagged_row in flagged_samples.iterrows():
        m_id = flagged_row['meeting_id']
        raw_row = df_eval[df_eval['meeting_id'] == m_id].iloc[0]
        transcript_excerpt = raw_row['transcript'][:600] + "..."
        reference = raw_row['reference_summary']
        candidate = raw_row['candidate_summary']
        
        report_content += f"""### Meeting ID: {m_id}
*   **BERTScore F1**: `{flagged_row['candidate_bertscore_f1']:.4f}`
*   **Factual Consistency**: `{flagged_row['candidate_factual_consistency']:.4f}`
*   **Action-Item F1**: `{flagged_row['candidate_action_item_f1']:.4f}`

**Transcript Excerpt**:
> {transcript_excerpt}

**Reference Summary**:
{reference}

**Candidate Summary**:
{candidate}

---
"""

    report_content += """
## Recommendations
- **Compare Lead-N and Candidate**: If Lead-N scores higher on factual consistency but lower on semantic similarity/ROUGE, it suggests the summarization model is paraphrasing well but occasionally introducing minor mis-statements.
- **Action Item Performance**: F1 score for action items indicates how accurately the model captures specific responsibilities compared to human editors. If recall is low but precision is high, the model is conservative (captures fewer action items, but what it does capture is correct).
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    latest_dir = "results/latest"
    if os.path.exists(latest_dir):
        if os.path.islink(latest_dir):
            os.unlink(latest_dir)
        else:
            shutil.rmtree(latest_dir)
            
    os.makedirs(latest_dir, exist_ok=True)
    shutil.copy(per_sample_path, os.path.join(latest_dir, "per_sample_scores.csv"))
    shutil.copy(agg_path, os.path.join(latest_dir, "aggregate_scores.json"))
    shutil.copy(report_path, os.path.join(latest_dir, "eval_report.md"))
    
    logging.info(f"Evaluation complete. Versioned run folder created: {run_dir}")
    logging.info(f"Results copied to: {latest_dir}/")
    print(f"\nEvaluation successfully finished. Results saved to {run_dir}/")

if __name__ == "__main__":
    main()
