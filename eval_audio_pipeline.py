import os
import re
import sys
import json
import time
import argparse
import logging
import shutil
import zipfile
import datetime
import io
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Import existing pipeline components
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# RemoteFile class implementing HTTP Range requests for on-demand ZIP file streaming
class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        r = requests.head(url, allow_redirects=True)
        r.raise_for_status()
        self.length = int(r.headers.get('Content-Length', 0))
        if self.length == 0:
            raise ValueError(f"Content-Length of remote file {url} is 0 or missing.")

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.length + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size == -1:
            size = self.length - self.position
        if self.position >= self.length:
            return b""
        end = self.position + size - 1
        if end >= self.length:
            end = self.length - 1
        headers = {'Range': f'bytes={self.position}-{end}'}
        r = requests.get(self.url, headers=headers)
        r.raise_for_status()
        data = r.content
        self.position += len(data)
        return data

    def readable(self):
        return True


def get_city_from_meeting_id(meeting_id: str) -> str:
    mid = meeting_id.lower()
    if 'longbeach' in mid:
        return 'LongBeach'
    elif 'boston' in mid:
        return 'Boston'
    elif 'seattle' in mid:
        return 'Seattle'
    elif 'alameda' in mid:
        return 'Alameda'
    elif 'denver' in mid:
        return 'Denver'
    elif 'kingcounty' in mid or 'king county' in mid:
        return 'KingCounty'
    else:
        raise ValueError(f"Unknown city for meeting ID: {meeting_id}")


def get_transcripts_zip_filename(city: str) -> str:
    if city.lower() == 'alameda':
        return 'Alameda/Alameda-transcripts-videolist.zip'
    elif city.lower() == 'denver':
        return 'Denver/Denver-transcripts-videolist.zip'
    elif city.lower() == 'kingcounty':
        return 'KingCounty/transcripts/transcripts.zip'
    elif city.lower() == 'longbeach':
        return 'LongBeach/transcripts/transcripts.zip'
    elif city.lower() == 'seattle':
        return 'Seattle/transcripts/transcripts.zip'
    elif city.lower() == 'boston':
        return 'Boston/transcripts/transcripts.zip'
    else:
        return f'{city}/transcripts/transcripts.zip'


def extract_words_from_transcript_json(transcript_data: dict, start_time_sec: float, end_time_sec: float) -> str:
    # 10,000,000 ticks per second scale factor in MeetingBank transcript JSONs
    start_ticks = start_time_sec * 10000000
    end_ticks = end_time_sec * 10000000
    
    extracted_words = []
    for segment in transcript_data.get('segments', []):
        for nbest in segment.get('nbest', []):
            for word in nbest.get('words', []):
                offset = word.get('offset', 0)
                if start_ticks <= offset <= end_ticks:
                    w_text = word.get('text', '')
                    if w_text:
                        extracted_words.append(w_text)
                        
    return " ".join(extracted_words)


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


def call_llm_json(prompt: str, model_name: str) -> list:
    """Same triple extraction LLM query helper as in eval_pipeline.py."""
    if model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            logging.warning("GEMINI_API_KEY not found. Action item LLM extraction skipped.")
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
            logging.error(f"Ollama API request failed for action extraction: {e}")
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
        return []


def extract_action_items_from_summary(summary_text: str, model_name: str) -> list:
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
    parser = argparse.ArgumentParser(description="Evaluate MOM Summarization Pipeline with Audio")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to evaluate (default: 5)")
    parser.add_argument("--model", type=str, default="phi3", choices=["phi3", "gemini"], help="Candidate generator model (default: phi3)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Action-item cosine similarity matching threshold (default: 0.7)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    # Verify API key
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        logging.error("Error: ASSEMBLYAI_API_KEY not found in .env. Transcription requires AssemblyAI.")
        sys.exit(1)

    # Directories setup
    cache_dir = "data/processed/audio_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load mappings and datasets
    mb_json_path = "data/MeetingBank.json"
    mapping_json_path = "data/audio_zip_mapping.json"
    
    if not os.path.exists(mb_json_path) or not os.path.exists(mapping_json_path):
        logging.error(f"Required mapping files {mb_json_path} or {mapping_json_path} are missing. Please run preprocessing first.")
        sys.exit(1)
        
    with open(mb_json_path) as f:
        mb_data = json.load(f)
    with open(mapping_json_path) as f:
        audio_zip_mapping = json.load(f)

    logging.info("Downloading huuuyeah/meetingbank text dataset splits...")
    dataset = load_dataset("huuuyeah/meetingbank", split="test")
    
    logging.info(f"Beginning audio-summarization end-to-end evaluation on {args.n_samples} samples...")
    
    evaluated_count = 0
    samples_data = []
    
    # Iterate through the test dataset rows
    for row in dataset:
        if evaluated_count >= args.n_samples:
            break
            
        uid = row['uid']
        ref_summary = row['summary']
        
        # Parse UID
        parts = uid.split('_')
        if len(parts) < 3:
            logging.warning(f"Row UID {uid} does not follow City_Date_Item format. Skipping.")
            continue
            
        meeting_id = "_".join(parts[:-1])
        item_id = parts[-1]
        
        # Lookup in MeetingBank.json
        if meeting_id not in mb_data or item_id not in mb_data[meeting_id]['itemInfo']:
            logging.warning(f"MeetingID {meeting_id} or ItemID {item_id} not found in MeetingBank.json. Skipping.")
            continue
            
        meeting_info = mb_data[meeting_id]
        item_info = meeting_info['itemInfo'][item_id]
        
        start_time = float(item_info['startTime'])
        end_time = float(item_info['endTime'])
        duration = float(item_info['duration'])
        
        # Audio file matching
        transcript_file = meeting_info.get('Transcripts', '')
        audio_file = transcript_file.replace('.transcript.json', '')
        
        if not audio_file:
            logging.warning(f"No audio file mapping in transcripts field for {meeting_id}. Skipping.")
            continue
            
        if audio_file not in audio_zip_mapping:
            logging.warning(f"Audio file {audio_file} not found in zip mapping. Skipping.")
            continue
            
        zip_path = audio_zip_mapping[audio_file]
        city = get_city_from_meeting_id(meeting_id)
        
        # Download and crop audio (with caching support)
        local_mp3_path = os.path.join(cache_dir, audio_file)
        
        try:
            if not os.path.exists(local_mp3_path):
                logging.info(f"[{evaluated_count+1}] Extracting {audio_file} remotely from {zip_path}...")
                url = f"https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/resolve/main/{zip_path}"
                rf = RemoteFile(url)
                with zipfile.ZipFile(rf) as zf:
                    # Find exact path inside the zip file
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
                logging.info(f"[{evaluated_count+1}] Found {audio_file} in local cache.")
                
            # Crop audio locally using librosa and soundfile
            temp_segment_wav = "data/processed/temp_segment.wav"
            os.makedirs(os.path.dirname(temp_segment_wav), exist_ok=True)
            
            logging.info(f"Cropping segment {item_id} from {start_time}s to {end_time}s...")
            import soundfile
            info = soundfile.info(local_mp3_path)
            sr = info.samplerate
            start_frame = int(start_time * sr)
            num_frames = int(duration * sr)
            y, sr = soundfile.read(local_mp3_path, start=start_frame, frames=num_frames)
            soundfile.write(temp_segment_wav, y, sr)
            
            # Download transcripts zip and reconstruct reference transcript
            transcripts_zip = get_transcripts_zip_filename(city)
            logging.info(f"Downloading transcripts zip for {city}: {transcripts_zip}...")
            trans_zip_path = hf_hub_download(repo_id="huuuyeah/MeetingBank_Audio", filename=transcripts_zip, repo_type="dataset")
            
            # Reconstruct reference transcript by matching segment times
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
                # Fallback to text dataset transcript column if extraction yields empty text
                logging.warning("Reconstructed transcript is empty. Falling back to dataset row transcript.")
                reference_transcript = row['transcript']
                
            # Run existing ASR pipeline
            logging.info("Transcribing audio segment via AssemblyAI...")
            start_asr = time.time()
            generated_transcript = transcribe(temp_segment_wav)
            asr_latency = time.time() - start_asr
            
            # Generate Candidate Summaries
            logging.info("Generating candidate summary from ASR transcript...")
            asr_summary = generate_summary(generated_transcript, model_name=args.model)
            
            logging.info("Generating candidate summary from gold (reference) transcript...")
            gold_summary = generate_summary(reference_transcript, model_name=args.model)
            
            # Transcription Quality Metrics
            wer = compute_wer(reference_transcript, generated_transcript)
            cer = compute_cer(reference_transcript, generated_transcript)
            n_wer = compute_normalized_wer(reference_transcript, generated_transcript)
            
            logging.info(f"ASR Metrics: WER={wer:.4f}, CER={cer:.4f}, Normalized WER={n_wer:.4f}")
            
            # Summarization Metrics for ASR-Summary
            asr_rouge = compute_rouge(ref_summary, asr_summary)
            asr_bleu = compute_bleu(ref_summary, asr_summary)
            asr_meteor = compute_meteor(ref_summary, asr_summary)
            asr_bert = compute_bertscore(ref_summary, asr_summary)
            asr_sem = compute_semantic_cosine(ref_summary, asr_summary)
            asr_comp = compute_compression_ratio(generated_transcript, asr_summary)
            
            res_fact_asr = compute_factual_consistency(generated_transcript, asr_summary)
            asr_fact = res_fact_asr["factual_consistency"]
            fact_method = res_fact_asr["method"]
            
            ref_triples = extract_action_items_from_summary(ref_summary, args.model)
            asr_triples = extract_action_items_from_summary(asr_summary, args.model)
            asr_ai = compute_action_item_metrics(ref_triples, asr_triples, threshold=args.threshold)
            
            # Summarization Metrics for Gold-Summary
            gold_rouge = compute_rouge(ref_summary, gold_summary)
            gold_bleu = compute_bleu(ref_summary, gold_summary)
            gold_meteor = compute_meteor(ref_summary, gold_summary)
            gold_bert = compute_bertscore(ref_summary, gold_summary)
            gold_sem = compute_semantic_cosine(ref_summary, gold_summary)
            gold_comp = compute_compression_ratio(reference_transcript, gold_summary)
            
            res_fact_gold = compute_factual_consistency(reference_transcript, gold_summary)
            gold_fact = res_fact_gold["factual_consistency"]
            
            gold_triples = extract_action_items_from_summary(gold_summary, args.model)
            gold_ai = compute_action_item_metrics(ref_triples, gold_triples, threshold=args.threshold)
            
            # Add to samples
            samples_data.append({
                "uid": uid,
                "meeting_id": meeting_id,
                "item_id": item_id,
                "reference_transcript": reference_transcript,
                "generated_transcript": generated_transcript,
                "reference_summary": ref_summary,
                "generated_summary_from_asr": asr_summary,
                "generated_summary_from_gold": gold_summary,
                "wer": wer,
                "cer": cer,
                "normalized_wer": n_wer,
                "asr_latency": asr_latency,
                
                # ASR summaries scores
                "asr_rouge1": asr_rouge["rouge1"],
                "asr_rouge2": asr_rouge["rouge2"],
                "asr_rougeL": asr_rouge["rougeL"],
                "asr_bleu": asr_bleu,
                "asr_meteor": asr_meteor,
                "asr_bertscore_f1": asr_bert["bertscore_f1"],
                "asr_semantic_cosine": asr_sem,
                "asr_factual_consistency": asr_fact,
                "asr_compression_ratio": asr_comp,
                "asr_action_item_f1": asr_ai["action_item_f1"],
                
                # Gold summaries scores
                "gold_rouge1": gold_rouge["rouge1"],
                "gold_rouge2": gold_rouge["rouge2"],
                "gold_rougeL": gold_rouge["rougeL"],
                "gold_bleu": gold_bleu,
                "gold_meteor": gold_meteor,
                "gold_bertscore_f1": gold_bert["bertscore_f1"],
                "gold_semantic_cosine": gold_sem,
                "gold_factual_consistency": gold_fact,
                "gold_compression_ratio": gold_comp,
                "gold_action_item_f1": gold_ai["action_item_f1"],
            })
            
            evaluated_count += 1
            
        except Exception as ex:
            logging.error(f"Failed processing sample {uid}: {ex}", exc_info=True)
            logging.warning("Row skipped due to failure.")
            continue
            
    if not samples_data:
        logging.error("No samples were successfully evaluated.")
        sys.exit(1)
        
    df_scores = pd.DataFrame(samples_data)
    
    # Calculate aggregates
    metrics_to_agg = [
        "rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore_f1", 
        "semantic_cosine", "factual_consistency", "compression_ratio", "action_item_f1"
    ]
    
    trans_metrics = ["wer", "cer", "normalized_wer"]
    
    aggregates = {
        "transcription": {},
        "asr_summary": {},
        "gold_summary": {},
        "delta": {}
    }
    
    # Transcription stats
    for m in trans_metrics:
        vals = df_scores[m].dropna()
        aggregates["transcription"][m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0
        }
        
    # Summarization and Deltas
    for m in metrics_to_agg:
        asr_col = f"asr_{m}"
        gold_col = f"gold_{m}"
        
        asr_vals = df_scores[asr_col].dropna()
        gold_vals = df_scores[gold_col].dropna()
        
        aggregates["asr_summary"][m] = {
            "mean": float(asr_vals.mean()),
            "std": float(asr_vals.std()) if len(asr_vals) > 1 else 0.0
        }
        
        aggregates["gold_summary"][m] = {
            "mean": float(gold_vals.mean()),
            "std": float(gold_vals.std()) if len(gold_vals) > 1 else 0.0
        }
        
        # delta = score(gold) - score(asr)
        delta_vals = df_scores[gold_col] - df_scores[asr_col]
        aggregates["delta"][m] = {
            "mean": float(delta_vals.mean()),
            "std": float(delta_vals.std()) if len(delta_vals) > 1 else 0.0
        }
        
    # Save outputs
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = f"results/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    per_sample_path = os.path.join(run_dir, "per_sample_scores.csv")
    agg_path = os.path.join(run_dir, "aggregate_scores.json")
    report_path = os.path.join(run_dir, "eval_report.md")
    
    df_scores.to_csv(per_sample_path, index=False)
    with open(agg_path, 'w') as f:
        json.dump(aggregates, f, indent=2)
        
    # Generate Markdown Report
    report_content = f"""# Minutes of Meeting (MOM) Audio Evaluation Report
**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Candidate Generator Model**: `{args.model}`
**Dataset**: MeetingBank Audio & Text (Evaluation Sample Size: {len(df_scores)})

---

## 1. Transcription Quality (ASR Performance)

WER and CER are computed against reconstructed reference transcripts from the meeting word-level alignments.

| Metric | Mean (Std) | Description |
|---|---|---|
| **Word Error Rate (WER)** | {aggregates['transcription']['wer']['mean']:.4f} ({aggregates['transcription']['wer']['std']:.4f}) | Errors (ins/del/sub) over reference word count |
| **Character Error Rate (CER)** | {aggregates['transcription']['cer']['mean']:.4f} ({aggregates['transcription']['cer']['std']:.4f}) | Character-level edit distance errors |
| **Normalized WER** | {aggregates['transcription']['normalized_wer']['mean']:.4f} ({aggregates['transcription']['normalized_wer']['std']:.4f}) | WER after lowercasing and stripping punctuation |

---

## 2. Summarization Quality

Summaries are generated from both ASR transcript and Gold (reference) transcript.

| Metric | Asr Summary Mean (Std) | Gold Summary Mean (Std) | Description |
|---|---|---|---|
| **ROUGE-1** | {aggregates['asr_summary']['rouge1']['mean']:.4f} ({aggregates['asr_summary']['rouge1']['std']:.4f}) | {aggregates['gold_summary']['rouge1']['mean']:.4f} ({aggregates['gold_summary']['rouge1']['std']:.4f}) | Word unigram overlap |
| **ROUGE-2** | {aggregates['asr_summary']['rouge2']['mean']:.4f} ({aggregates['asr_summary']['rouge2']['std']:.4f}) | {aggregates['gold_summary']['rouge2']['mean']:.4f} ({aggregates['gold_summary']['rouge2']['std']:.4f}) | Word bigram overlap |
| **ROUGE-L** | {aggregates['asr_summary']['rougeL']['mean']:.4f} ({aggregates['asr_summary']['rougeL']['std']:.4f}) | {aggregates['gold_summary']['rougeL']['mean']:.4f} ({aggregates['gold_summary']['rougeL']['std']:.4f}) | Longest Common Subsequence |
| **BLEU** | {aggregates['asr_summary']['bleu']['mean']:.4f} ({aggregates['asr_summary']['bleu']['std']:.4f}) | {aggregates['gold_summary']['bleu']['mean']:.4f} ({aggregates['gold_summary']['bleu']['std']:.4f}) | Sentence translation match |
| **METEOR** | {aggregates['asr_summary']['meteor']['mean']:.4f} ({aggregates['asr_summary']['meteor']['std']:.4f}) | {aggregates['gold_summary']['meteor']['mean']:.4f} ({aggregates['gold_summary']['meteor']['std']:.4f}) | Paraphrase-aware overlap |
| **BERTScore F1** | {aggregates['asr_summary']['bertscore_f1']['mean']:.4f} ({aggregates['asr_summary']['bertscore_f1']['std']:.4f}) | {aggregates['gold_summary']['bertscore_f1']['mean']:.4f} ({aggregates['gold_summary']['bertscore_f1']['std']:.4f}) | Roberta semantic similarity |
| **Embedding Cosine** | {aggregates['asr_summary']['semantic_cosine']['mean']:.4f} ({aggregates['asr_summary']['semantic_cosine']['std']:.4f}) | {aggregates['gold_summary']['semantic_cosine']['mean']:.4f} ({aggregates['gold_summary']['semantic_cosine']['std']:.4f}) | MiniLM sentence cosine similarity |
| **Factual Consistency** | {aggregates['asr_summary']['factual_consistency']['mean']:.4f} ({aggregates['asr_summary']['factual_consistency']['std']:.4f}) | {aggregates['gold_summary']['factual_consistency']['mean']:.4f} ({aggregates['gold_summary']['factual_consistency']['std']:.4f}) | SummaC / NLI entailment score |
| **Action-Item F1** | {aggregates['asr_summary']['action_item_f1']['mean']:.4f} ({aggregates['asr_summary']['action_item_f1']['std']:.4f}) | {aggregates['gold_summary']['action_item_f1']['mean']:.4f} ({aggregates['gold_summary']['action_item_f1']['std']:.4f}) | Match of action/owner/deadline triples |
| **Compression Ratio** | {aggregates['asr_summary']['compression_ratio']['mean']:.4f} ({aggregates['asr_summary']['compression_ratio']['std']:.4f}) | {aggregates['gold_summary']['compression_ratio']['mean']:.4f} ({aggregates['gold_summary']['compression_ratio']['std']:.4f}) | Source length / summary length |

---

## 3. Error Propagation Analysis

The delta measures quality degradation directly attributable to ASR (transcription) errors:
`delta = score(gold_summary) - score(asr_summary)`

A larger delta indicates that transcription errors have degraded downstream summarization quality. A delta close to zero suggests the summarizer model is robust to ASR noise.

| Metric | Delta Mean (Std) | Interpretation |
|---|---|---|
| **ROUGE-1 Delta** | {aggregates['delta']['rouge1']['mean']:.4f} ({aggregates['delta']['rouge1']['std']:.4f}) | Loss in unigram recall |
| **ROUGE-2 Delta** | {aggregates['delta']['rouge2']['mean']:.4f} ({aggregates['delta']['rouge2']['std']:.4f}) | Loss in bigram structure |
| **ROUGE-L Delta** | {aggregates['delta']['rougeL']['mean']:.4f} ({aggregates['delta']['rougeL']['std']:.4f}) | Loss in sentence-level sequence |
| **BERTScore F1 Delta** | {aggregates['delta']['bertscore_f1']['mean']:.4f} ({aggregates['delta']['bertscore_f1']['std']:.4f}) | Loss in semantic alignment |
| **Embedding Cosine Delta** | {aggregates['delta']['semantic_cosine']['mean']:.4f} ({aggregates['delta']['semantic_cosine']['std']:.4f}) | Overall semantic distance degradation |
| **Factual Consistency Delta** | {aggregates['delta']['factual_consistency']['mean']:.4f} ({aggregates['delta']['factual_consistency']['std']:.4f}) | Impact of ASR errors on factual accuracy |
| **Action-Item F1 Delta** | {aggregates['delta']['action_item_f1']['mean']:.4f} ({aggregates['delta']['action_item_f1']['std']:.4f}) | Loss in actionable task retrieval |

---

## 4. Manual Verification & Auditing (First Evaluated Sample)

To manually listen to the cropped segment audio and verify its alignment with the reconstructed reference transcript:

*   **Sample UID**: `{df_scores.iloc[0]['uid']}`
*   **Audio Path**: `data/processed/audio_verify_{df_scores.iloc[0]['item_id']}.wav`

**Reconstructed Reference Transcript**:
> {df_scores.iloc[0]['reference_transcript'][:800]}...

**ASR Generated Transcript**:
> {df_scores.iloc[0]['generated_transcript'][:800]}...
"""
    # Write manual verification audio file for review
    first_row = df_scores.iloc[0]
    manual_verify_audio = f"data/processed/audio_verify_{first_row['item_id']}.wav"
    shutil.copy("data/processed/temp_segment.wav", manual_verify_audio)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
        
    # Copy to latest
    latest_dir = "results/latest"
    os.makedirs(latest_dir, exist_ok=True)
    shutil.copy(per_sample_path, os.path.join(latest_dir, "per_sample_scores.csv"))
    shutil.copy(agg_path, os.path.join(latest_dir, "aggregate_scores.json"))
    shutil.copy(report_path, os.path.join(latest_dir, "eval_report.md"))
    
    logging.info(f"Audio evaluation complete. Versioned run folder created: {run_dir}")
    print(f"\nEvaluation successfully finished. Results saved to {run_dir}/")
    print(f"Manual verification audio copy written to: {manual_verify_audio}")


if __name__ == "__main__":
    main()
