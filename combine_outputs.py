import os
import json

def main():
    chunks_dir = "chunks"
    combined_transcript_path = "all_chunks_diarized_transcript.txt"
    combined_summary_path = "all_chunks_summaries.txt"
    
    # 1. Combine Transcripts
    combined_transcript = []
    for i in range(1, 6):
        tx_file = os.path.join(chunks_dir, f"chunk_{i}_transcript.txt")
        if os.path.exists(tx_file):
            with open(tx_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            combined_transcript.append(f"=== Chunk {i} ===\n{content}\n")
        else:
            print(f"Warning: {tx_file} not found.")
            
    transcript_text = "\n".join(combined_transcript)
    
    # Load and format transcription_evaluation.json
    trans_eval_content = ""
    if os.path.exists("transcription_evaluation.json"):
        with open("transcription_evaluation.json", "r", encoding="utf-8") as f:
            t_eval = json.load(f)
        trans_eval_content += "\n\n=========================================\n"
        trans_eval_content += "        TRANSCRIPTION EVALUATION         \n"
        trans_eval_content += "=========================================\n"
        metadata = t_eval.get("metadata", {})
        trans_eval_content += f"Average Word Error Rate (WER): {metadata.get('average_word_error_rate', 0.0):.2%}\n"
        trans_eval_content += f"Average STT Accuracy: {metadata.get('average_accuracy', 0.0):.2%}\n\n"
        
        for uid, eval_item in t_eval.get("evaluations", {}).items():
            trans_eval_content += f"UID: {uid}\n"
            trans_eval_content += f"  Word Error Rate (WER): {eval_item.get('word_error_rate', 0.0):.2%}\n"
            trans_eval_content += f"  STT Accuracy: {eval_item.get('accuracy', 0.0):.2%}\n"
            trans_eval_content += "-" * 40 + "\n"
            
    with open(combined_transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text + trans_eval_content)
    print(f"Combined diarized transcripts and evaluations saved to: {combined_transcript_path}")
    
    # 2. Combine Summaries
    combined_summary = []
    for i in range(1, 6):
        sum_file = os.path.join(chunks_dir, f"chunk_{i}_summary.txt")
        if os.path.exists(sum_file):
            with open(sum_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Clean summary from any self-eval headers if present
            clean_content = content.split("\n\n=== AI Self-Evaluation ===")[0].strip()
            combined_summary.append(f"=== Chunk {i} Summary & Action Items ===\n{clean_content}\n")
        else:
            print(f"Warning: {sum_file} not found.")
            
    summary_text = "\n".join(combined_summary)
    
    # Load and format summarization_evaluation.json
    sum_eval_content = ""
    if os.path.exists("summarization_evaluation.json"):
        with open("summarization_evaluation.json", "r", encoding="utf-8") as f:
            s_eval = json.load(f)
        sum_eval_content += "\n\n=========================================\n"
        sum_eval_content += "        SUMMARIZATION EVALUATION         \n"
        sum_eval_content += "=========================================\n"
        metadata = s_eval.get("metadata", {})
        sum_eval_content += f"Average ROUGE-1 F1: {metadata.get('average_rouge1', 0.0):.2%}\n"
        sum_eval_content += f"Average ROUGE-2 F1: {metadata.get('average_rouge2', 0.0):.2%}\n"
        sum_eval_content += f"Average ROUGE-L F1: {metadata.get('average_rougeL', 0.0):.2%}\n"
        sum_eval_content += f"Average LLM Judge Score: {metadata.get('average_llm_judge_score', 0.0):.1f}/10\n\n"
        
        for uid, eval_item in s_eval.get("evaluations", {}).items():
            sum_eval_content += f"UID: {uid}\n"
            r_scores = eval_item.get("rouge_scores", {})
            sum_eval_content += f"  ROUGE-1 F1: {r_scores.get('rouge1', {}).get('fmeasure', 0.0):.2%}\n"
            sum_eval_content += f"  ROUGE-2 F1: {r_scores.get('rouge2', {}).get('fmeasure', 0.0):.2%}\n"
            sum_eval_content += f"  ROUGE-L F1: {r_scores.get('rougeL', {}).get('fmeasure', 0.0):.2%}\n"
            llm_judge = eval_item.get("llm_judge", {})
            sum_eval_content += f"  LLM Judge Score: {llm_judge.get('score', 0)}/10\n"
            sum_eval_content += f"  LLM Judge Feedback: {llm_judge.get('feedback', '').strip()}\n"
            sum_eval_content += "-" * 40 + "\n"
            
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + sum_eval_content)
    print(f"Combined summaries and evaluations saved to: {combined_summary_path}")

if __name__ == "__main__":
    main()
