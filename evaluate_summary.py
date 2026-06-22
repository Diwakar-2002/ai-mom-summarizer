import os
import json
import re
import requests
from rouge_score import rouge_scorer
from summarize import generate_summary

def get_llm_judge_score(gt_summary, gen_summary):
    eval_prompt = (
        "You are an expert evaluator. Compare the following generated summary with the ground truth summary.\n"
        "Rate the generated summary on a scale of 1-10 based on:\n"
        "1. Factual Correctness\n"
        "2. Completeness (does it capture all main points?)\n"
        "3. Clarity and Conciseness\n\n"
        "Provide the score in the format 'Score: X/10' where X is an integer, and a brief 1-2 sentence justification.\n\n"
        f"Ground Truth Summary:\n{gt_summary}\n\n"
        f"Generated Summary:\n{gen_summary}"
    )
    
    payload = {
        "model": "phi3",
        "prompt": eval_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "top_p": 0.9,
            "num_predict": 150
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        
        # Extract the score using robust regex matching
        score = 0
        text_lower = result.lower()
        
        # 1. Match "score: X/10" or "score: X / 10"
        m1 = re.search(r"score:\s*(\d+)\s*/\s*10", text_lower)
        if m1:
            score = int(m1.group(1))
        else:
            # 2. Match "X/10"
            m2 = re.search(r"(\d+)\s*/\s*10", text_lower)
            if m2:
                score = int(m2.group(1))
            else:
                # 3. Match "score: X/1" (mangled /10)
                m3 = re.search(r"score:\s*(\d+)\s*/\s*1", text_lower)
                if m3:
                    score = int(m3.group(1))
                else:
                    # 4. Match "score: X"
                    m4 = re.search(r"score:\s*(\d+)", text_lower)
                    if m4:
                        score = int(m4.group(1))
                    else:
                        # 5. Find first number after "score"
                        m5 = re.search(r"score.*?(\d+)", text_lower, re.DOTALL)
                        if m5:
                            score = int(m5.group(1))
                            
        # Ensure score is within valid bounds [1, 10]
        if score < 1 or score > 10:
            score = 0
            
        return score, result
    except Exception as e:
        print(f"Error calling Ollama for LLM judge: {e}", flush=True)
        return 0, f"Error: {e}"

def main():
    # Step 1: Load aligned transcripts
    aligned_path = "aligned_transcripts.json"
    if not os.path.exists(aligned_path):
        print(f"Error: '{aligned_path}' not found. Please run evaluate_stt.py first.", flush=True)
        return
        
    with open(aligned_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)
        
    # Step 2: Load transcription evaluations to get chunk transcripts
    stt_eval_path = "transcription_evaluation.json"
    if not os.path.exists(stt_eval_path):
        print(f"Error: '{stt_eval_path}' not found. Please run evaluate_stt.py first.", flush=True)
        return
        
    with open(stt_eval_path, 'r', encoding='utf-8') as f:
        stt_eval_data = json.load(f)
        
    # Step 3: Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    evaluation_results = {
        "metadata": {
            "evaluation_type": "summarization",
            "source_audio": "alameda_05fd2fe2-ce9f-48cf-8acc-c0a49d6a8067.mp3",
            "llm_model": "phi3"
        },
        "chunks": {},
        "evaluations": {}
    }
    
    # Generate and save summaries for the 5 chunks
    print("\n--- Generating Summaries for Chunks ---", flush=True)
    for chunk_key, chunk_info in stt_eval_data["chunks"].items():
        chunk_text = chunk_info["text"]
        print(f"Summarizing {chunk_key}...", flush=True)
        
        if not chunk_text.strip():
            print(f"  Warning: {chunk_key} transcript is empty. Skipping.", flush=True)
            summary_full = "Empty transcript."
        else:
            summary_full = generate_summary(chunk_text, meeting_type="general", skip_eval=True, max_words=600)
            
        # Clean the summary to get only the MOM text (exclude AI self-evaluation)
        summary_clean = summary_full.split("\n\n=== AI Self-Evaluation ===")[0].strip()
        
        # Save chunk summary to txt
        summary_file = os.path.join("chunks", f"{chunk_key}_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_full)
            
        evaluation_results["chunks"][chunk_key] = {
            "summary_file": summary_file,
            "summary_text": summary_clean,
            "full_summary_with_self_eval": summary_full
        }
        print(f"  Saved summary to: {summary_file}", flush=True)
        
    # Evaluate summarization against Hugging Face ground truth
    print("\n--- Evaluating Summarization against Ground Truth (Hugging Face) ---", flush=True)
    total_r1 = 0.0
    total_r2 = 0.0
    total_rl = 0.0
    total_judge = 0.0
    num_evals = 0
    
    for uid, data in aligned_data.items():
        matched_transcript = data["matched_transcript"]
        gt_summary = data["ground_truth_summary"]
        
        print(f"Evaluating UID: {uid}...", flush=True)
        
        if not matched_transcript.strip():
            print(f"  Warning: Matched transcript for {uid} is empty. Skipping.", flush=True)
            continue
            
        # Generate summary for the matched transcript segment
        summary_full = generate_summary(matched_transcript, meeting_type="general", skip_eval=True, max_words=600)
        summary_clean = summary_full.split("\n\n=== AI Self-Evaluation ===")[0].strip()
        
        # Calculate ROUGE scores
        scores = scorer.score(gt_summary, summary_clean)
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure
        
        # Calculate LLM-as-a-judge score
        judge_score, judge_feedback = get_llm_judge_score(gt_summary, summary_clean)
        
        total_r1 += r1
        total_r2 += r2
        total_rl += rl
        total_judge += judge_score
        num_evals += 1
        
        print(f"  ROUGE-1 F1: {r1:.2%}", flush=True)
        print(f"  ROUGE-2 F1: {r2:.2%}", flush=True)
        print(f"  ROUGE-L F1: {rl:.2%}", flush=True)
        print(f"  LLM Judge Score: {judge_score}/10", flush=True)
        print(f"  Feedback: {judge_feedback}", flush=True)
        print("-" * 45, flush=True)
        
        evaluation_results["evaluations"][uid] = {
            "ground_truth_summary": gt_summary,
            "generated_summary": summary_clean,
            "full_summary_with_self_eval": summary_full,
            "rouge_scores": {
                "rouge1": {"precision": scores['rouge1'].precision, "recall": scores['rouge1'].recall, "fmeasure": r1},
                "rouge2": {"precision": scores['rouge2'].precision, "recall": scores['rouge2'].recall, "fmeasure": r2},
                "rougeL": {"precision": scores['rougeL'].precision, "recall": scores['rougeL'].recall, "fmeasure": rl}
            },
            "llm_judge": {
                "score": judge_score,
                "feedback": judge_feedback
            }
        }
        
    if num_evals > 0:
        avg_r1 = total_r1 / num_evals
        avg_r2 = total_r2 / num_evals
        avg_rl = total_rl / num_evals
        avg_judge = total_judge / num_evals
        
        evaluation_results["metadata"]["average_rouge1"] = avg_r1
        evaluation_results["metadata"]["average_rouge2"] = avg_r2
        evaluation_results["metadata"]["average_rougeL"] = avg_rl
        evaluation_results["metadata"]["average_llm_judge_score"] = avg_judge
        
        print(f"Average ROUGE-1 F1: {avg_r1:.2%}", flush=True)
        print(f"Average ROUGE-2 F1: {avg_r2:.2%}", flush=True)
        print(f"Average ROUGE-L F1: {avg_rl:.2%}", flush=True)
        print(f"Average LLM Judge Score: {avg_judge:.1f}/10", flush=True)
    else:
        print("No evaluations performed.", flush=True)
        
    # Save results to summarization_evaluation.json
    with open("summarization_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print("\nSummarization evaluations saved to 'summarization_evaluation.json'", flush=True)

    # Automatically generate combined outputs and append evaluations
    try:
        import combine_outputs
        print("\nGenerating master combined output files...", flush=True)
        combine_outputs.main()
    except Exception as e:
        print(f"Error generating combined master files: {e}", flush=True)

if __name__ == "__main__":
    main()
