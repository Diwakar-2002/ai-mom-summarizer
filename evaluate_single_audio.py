import os
import re
import sys
import json
import time
import zipfile
import logging
import shutil
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import soundfile

# Load dotenv
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Import pipeline helper logic from existing files
from eval_audio_pipeline import (
    RemoteFile, 
    get_city_from_meeting_id, 
    get_transcripts_zip_filename, 
    extract_words_from_transcript_json
)
from summarize import generate_summary
from metrics import (
    compute_rouge,
    compute_bleu,
    compute_meteor,
    compute_bertscore,
    compute_semantic_cosine,
    compute_factual_consistency,
    compute_compression_ratio,
    compute_action_item_metrics,
    compute_wer,
    compute_cer,
    compute_normalized_wer
)

def transcribe(audio_path: str) -> str:
    """Thin wrapper around AssemblyAI calling universal-2 speaker labels model."""
    import assemblyai as aai
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY environment variable is not set.")
    
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=True
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)
    
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
        
    full_transcript_text = ""
    if getattr(transcript, 'utterances', None):
        for utterance in transcript.utterances:
            full_transcript_text += f"Speaker {utterance.speaker}: {utterance.text}\n"
    else:
        full_transcript_text = transcript.text
        
    return full_transcript_text.strip()

def extract_action_items_from_summary(summary_text: str, model_name: str) -> list:
    """Helper to query the LLM for action item extraction."""
    from eval_audio_pipeline import call_llm_json
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

def main():
    target_uid = "SeattleCityCouncil_06132016_Res 31669"
    logging.info(f"Targeting sample: {target_uid}")
    
    # 1. Load Hugging Face dataset split and get ground truth summary & details
    logging.info("Loading huuuyeah/meetingbank text dataset splits...")
    dataset = load_dataset("huuuyeah/meetingbank", split="test")
    
    target_row = None
    for row in dataset:
        if row['uid'] == target_uid:
            target_row = row
            break
            
    if not target_row:
        logging.error(f"Failed to find test sample with UID: {target_uid}")
        sys.exit(1)
        
    ref_summary = target_row['summary']
    logging.info("Found target test sample and reference summary.")
    
    # Parse meeting ID and item ID
    parts = target_uid.split('_')
    meeting_id = "_".join(parts[:-1])
    item_id = parts[-1]
    
    # Load mappings
    mb_json_path = "data/MeetingBank.json"
    mapping_json_path = "data/audio_zip_mapping.json"
    
    with open(mb_json_path) as f:
        mb_data = json.load(f)
    with open(mapping_json_path) as f:
        audio_zip_mapping = json.load(f)
        
    if meeting_id not in mb_data or item_id not in mb_data[meeting_id]['itemInfo']:
        logging.error("Target meeting or item not found in MeetingBank.json.")
        sys.exit(1)
        
    meeting_info = mb_data[meeting_id]
    item_info = meeting_info['itemInfo'][item_id]
    
    start_time = float(item_info['startTime'])
    end_time = float(item_info['endTime'])
    duration = float(item_info['duration'])
    
    transcript_file = meeting_info.get('Transcripts', '')
    audio_file = transcript_file.replace('.transcript.json', '')
    zip_path = audio_zip_mapping[audio_file]
    city = get_city_from_meeting_id(meeting_id)
    
    # 2. Check or download raw audio
    cache_dir = "data/processed/audio_cache"
    local_mp3_path = os.path.join(cache_dir, audio_file)
    
    if not os.path.exists(local_mp3_path):
        logging.info(f"Audio file not found in local cache. Downloading {audio_file} remotely from {zip_path}...")
        os.makedirs(cache_dir, exist_ok=True)
        url = f"https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/resolve/main/{zip_path}"
        rf = RemoteFile(url)
        with zipfile.ZipFile(rf) as zf:
            zip_name = None
            for name in zf.namelist():
                if os.path.basename(name) == audio_file:
                    zip_name = name
                    break
            if not zip_name:
                raise FileNotFoundError(f"Audio file {audio_file} not found inside zip {zip_path}")
            with open(local_mp3_path, 'wb') as out_f:
                out_f.write(zf.read(zip_name))
        logging.info(f"Saved to cache: {local_mp3_path}")
    else:
        logging.info(f"Found raw audio file {audio_file} in cache.")
        
    # 3. Crop audio locally using soundfile and save directly to the workspace root directory
    output_audio_path = "meeting_segment_Res_31669.wav"
    logging.info(f"Cropping segment {item_id} from {start_time}s to {end_time}s...")
    
    info = soundfile.info(local_mp3_path)
    sr = info.samplerate
    start_frame = int(start_time * sr)
    num_frames = int(duration * sr)
    y, sr = soundfile.read(local_mp3_path, start=start_frame, frames=num_frames)
    soundfile.write(output_audio_path, y, sr)
    logging.info(f"Saved cropped audio to workspace root: {output_audio_path}")
    
    # 4. Download and extract ground-truth transcript
    transcripts_zip = get_transcripts_zip_filename(city)
    logging.info(f"Downloading transcripts zip for {city} to HF cache: {transcripts_zip}...")
    trans_zip_path = hf_hub_download(repo_id="huuuyeah/MeetingBank_Audio", filename=transcripts_zip, repo_type="dataset")
    
    with zipfile.ZipFile(trans_zip_path, 'r') as tz:
        tz_name = None
        for name in tz.namelist():
            if os.path.basename(name) == transcript_file:
                tz_name = name
                break
        if not tz_name:
            raise FileNotFoundError(f"Transcript JSON {transcript_file} not found in transcripts zip.")
        transcript_data = json.loads(tz.read(tz_name))
        
    reference_transcript = extract_words_from_transcript_json(transcript_data, start_time, end_time)
    if not reference_transcript.strip():
        logging.warning("Reconstructed reference transcript is empty. Falling back to dataset row transcript.")
        reference_transcript = target_row['transcript']
        
    # 5. Generate transcription done by my model (AssemblyAI)
    logging.info("Transcribing audio segment via AssemblyAI...")
    start_time_asr = time.time()
    generated_transcript = transcribe(output_audio_path)
    asr_latency = time.time() - start_time_asr
    logging.info(f"Transcription finished in {asr_latency:.2f} seconds.")
    
    # 6. Generate summarization from my model (Ollama phi3)
    logging.info("Generating candidate summary from generated transcript...")
    model_name = "phi3"
    asr_summary = generate_summary(generated_transcript, model_name=model_name)
    logging.info("Summary generated successfully.")
    
    # 7. Calculate evaluation metrics
    logging.info("Calculating evaluation metrics...")
    
    # A. Transcription Metrics
    wer = compute_wer(reference_transcript, generated_transcript)
    cer = compute_cer(reference_transcript, generated_transcript)
    n_wer = compute_normalized_wer(reference_transcript, generated_transcript)
    
    # B. Summarization Metrics
    asr_rouge = compute_rouge(ref_summary, asr_summary)
    asr_bleu = compute_bleu(ref_summary, asr_summary)
    asr_meteor = compute_meteor(ref_summary, asr_summary)
    asr_bert = compute_bertscore(ref_summary, asr_summary)
    asr_sem = compute_semantic_cosine(ref_summary, asr_summary)
    asr_comp = compute_compression_ratio(generated_transcript, asr_summary)
    
    res_fact_asr = compute_factual_consistency(generated_transcript, asr_summary)
    asr_fact = res_fact_asr["factual_consistency"]
    fact_method = res_fact_asr["method"]
    
    ref_triples = extract_action_items_from_summary(ref_summary, model_name)
    asr_triples = extract_action_items_from_summary(asr_summary, model_name)
    asr_ai = compute_action_item_metrics(ref_triples, asr_triples, threshold=0.7)
    
    metrics = {
        "transcription": {
            "wer": wer,
            "cer": cer,
            "normalized_wer": n_wer,
            "latency_seconds": asr_latency
        },
        "summarization": {
            "rouge1": asr_rouge["rouge1"],
            "rouge2": asr_rouge["rouge2"],
            "rougeL": asr_rouge["rougeL"],
            "bleu": asr_bleu,
            "meteor": asr_meteor,
            "bertscore_f1": asr_bert["bertscore_f1"],
            "semantic_cosine": asr_sem,
            "factual_consistency": asr_fact,
            "factual_method": fact_method,
            "compression_ratio": asr_comp,
            "action_item_precision": asr_ai["action_item_precision"],
            "action_item_recall": asr_ai["action_item_recall"],
            "action_item_f1": asr_ai["action_item_f1"]
        }
    }
    
    # 8. Save structured JSON result file in root directory
    output_json_path = "single_evaluation_report.json"
    with open(output_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved structured JSON metrics to root: {output_json_path}")
    
    # 9. Save Markdown Report file in root directory
    output_report_path = "single_evaluation_report.md"
    
    report_content = f"""# MOM Single Audio Clip Evaluation Report
**Target Sample UID**: `{target_uid}`
**Date Evaluated**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Candidate Generator Model**: `{model_name}`
**ASR Model**: `AssemblyAI Universal-2`

---

## 1. Calculated Evaluation Metrics

### Transcription Quality (ASR vs. Ground Truth)
| Metric | Value | Description |
|---|---|---|
| **Word Error Rate (WER)** | {wer:.4f} | Edit distance over reference word count |
| **Character Error Rate (CER)** | {cer:.4f} | Character-level edit distance |
| **Normalized WER** | {n_wer:.4f} | WER after lowercasing and stripping punctuation |
| **Transcription Latency** | {asr_latency:.2f}s | Latency to upload and transcribe segment |

### Summarization Quality (Model Summary vs. Ground Truth)
| Metric | Value | Description |
|---|---|---|
| **ROUGE-1** | {asr_rouge['rouge1']:.4f} | Unigram lexical overlap |
| **ROUGE-2** | {asr_rouge['rouge2']:.4f} | Bigram lexical overlap |
| **ROUGE-L** | {asr_rouge['rougeL']:.4f} | Longest Common Subsequence |
| **BLEU** | {asr_bleu:.4f} | Sentence translation match |
| **METEOR** | {asr_meteor:.4f} | Paraphrase-aware overlap |
| **BERTScore F1** | {asr_bert['bertscore_f1']:.4f} | RoBERTa semantic similarity |
| **Embedding Cosine** | {asr_sem:.4f} | Sentence Embedding Cosine Similarity |
| **Factual Consistency** | {asr_fact:.4f} | Factual grounding score (Method: `{fact_method}`) |
| **Action-Item Precision** | {asr_ai['action_item_precision']:.4f} | Precision of matched action/owner/deadline triples |
| **Action-Item Recall** | {asr_ai['action_item_recall']:.4f} | Recall of matched action/owner/deadline triples |
| **Action-Item F1** | {asr_ai['action_item_f1']:.4f} | Overall F1 score for action items (Sim Threshold: `0.7`) |
| **Compression Ratio** | {asr_comp:.4f} | Source transcript length / Generated summary length |

---

## 2. Transcription Comparison

### Ground Truth (Reference) Transcript (First 800 chars)
> {reference_transcript[:800]}...

### Model (AssemblyAI) Transcript (First 800 chars)
> {generated_transcript[:800]}...

---

## 3. Summarization Comparison

### Ground Truth (Reference) Summary
{ref_summary}

### Model (Ollama/Phi3) Summary
{asr_summary}

---

## 4. Full Transcripts

### Full Ground Truth (Reference) Transcript
{reference_transcript}

### Full Model (AssemblyAI) Transcript
{generated_transcript}
"""
    with open(output_report_path, 'w') as f:
        f.write(report_content)
        
    logging.info(f"Saved Markdown report to root: {output_report_path}")
    print("\n=======================================================")
    print("Single Audio Clip Evaluation Completed Successfully!")
    print(f"1. Audio Segment: {output_audio_path}")
    print(f"2. Structured JSON Metrics: {output_json_path}")
    print(f"3. Markdown Report: {output_report_path}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
