import os
import json
import re
import argparse
from dotenv import load_dotenv

try:
    import jiwer
except ImportError:
    jiwer = None

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def find_best_alignment(local_words, gt_words):
    n_gt = len(gt_words)
    if n_gt == 0 or len(local_words) == 0:
        return "", 1.0, 0, 0
        
    best_overlap = -1
    best_idx = 0
    gt_set = set(gt_words)
    
    # Fast bag-of-words sliding window search
    step = max(1, n_gt // 20)
    for i in range(0, len(local_words) - n_gt + 1, step):
        window = local_words[i:i+n_gt]
        overlap = len(gt_set.intersection(window))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
            
    # Fine-tune boundaries using WER
    best_wer = 999.0
    best_segment_words = []
    best_start = best_idx
    best_end = best_idx + n_gt
    
    search_range = 60
    gt_text = " ".join(gt_words)
    
    for start_offset in range(-search_range, search_range + 1, 10):
        start = max(0, best_idx + start_offset)
        for len_offset in range(-search_range, search_range + 1, 10):
            end = min(len(local_words), start + n_gt + len_offset)
            if end <= start:
                continue
            candidate = local_words[start:end]
            cand_text = " ".join(candidate)
            if not cand_text:
                continue
            err = jiwer.wer(gt_text, cand_text)
            if err < best_wer:
                best_wer = err
                best_segment_words = candidate
                best_start = start
                best_end = end
                
    return " ".join(best_segment_words), best_wer, best_start, best_end

def extract_offline_transcripts(json_path):
    print(f"Extracting chunk transcripts offline from '{json_path}' with speaker diarization...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    chunks = {i: [] for i in range(5)}
    chunk_duration = 900  # 15 minutes in seconds
    
    for seg in data['segments']:
        offset_sec = seg['offset'] / 1e7
        # Find which 15-minute chunk this segment belongs to
        chunk_idx = int(offset_sec // chunk_duration)
        if 0 <= chunk_idx < 5:
            text = seg['nbest'][0]['text']
            speaker = seg.get('speaker', None)
            chunks[chunk_idx].append({'speaker': speaker, 'text': text})
            
    chunk_diarized = {}
    chunk_raw = {}
    
    for i in range(5):
        raw_texts = []
        diarized_lines = []
        current_speaker = None
        current_speaker_text = []
        
        for item in chunks[i]:
            raw_texts.append(item['text'])
            speaker = item['speaker']
            
            # Map speaker index to letters (e.g. 0 -> A, 1 -> B)
            if speaker is not None:
                speaker_label = chr(ord('A') + speaker)
            else:
                speaker_label = None
                
            if speaker_label != current_speaker:
                if current_speaker is not None and current_speaker_text:
                    diarized_lines.append(f"Speaker {current_speaker}: {' '.join(current_speaker_text)}")
                current_speaker = speaker_label
                current_speaker_text = [item['text']]
            else:
                current_speaker_text.append(item['text'])
                
        if current_speaker is not None and current_speaker_text:
            diarized_lines.append(f"Speaker {current_speaker}: {' '.join(current_speaker_text)}")
            
        chunk_raw[i+1] = " ".join(raw_texts)
        chunk_diarized[i+1] = "\n".join(diarized_lines) if diarized_lines else " ".join(raw_texts)
        
    return chunk_diarized, chunk_raw

def transcribe_live_chunks(api_key):
    print("Transcribing chunks using AssemblyAI API (Live with Speaker Diarization)...")
    import assemblyai as aai
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=True  # Enables speaker labels
    )
    transcriber = aai.Transcriber(config=config)
    
    chunk_diarized = {}
    chunk_raw = {}
    for i in range(1, 6):
        file_path = os.path.join("chunks", f"chunk_{i}.mp3")
        print(f"Uploading and transcribing '{file_path}'...")
        transcript = transcriber.transcribe(file_path)
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Error transcribing chunk {i}: {transcript.error}")
            chunk_diarized[i] = ""
            chunk_raw[i] = ""
        else:
            chunk_raw[i] = transcript.text
            
            # Format text with speaker labels
            if getattr(transcript, "utterances", None):
                diarized_lines = []
                for utterance in transcript.utterances:
                    diarized_lines.append(f"Speaker {utterance.speaker}: {utterance.text}")
                chunk_diarized[i] = "\n".join(diarized_lines)
            else:
                chunk_diarized[i] = transcript.text
            
    return chunk_diarized, chunk_raw

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate Speech-to-Text for 15-minute audio chunks.")
    parser.add_argument("--live", action="store_true", help="Perform live transcription using AssemblyAI")
    args = parser.parse_args()
    
    # Check if jiwer is installed
    if not jiwer:
        print("Error: 'jiwer' library is not installed. Please run 'pip install jiwer'.")
        return
        
    # Step 1: Transcribe the 5 chunks (either live or offline extraction)
    chunk_transcripts_diarized = {}
    chunk_transcripts_raw = {}
    if args.live:
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            print("Error: ASSEMBLYAI_API_KEY not found in environment variables. Falling back to offline extraction.")
            chunk_transcripts_diarized, chunk_transcripts_raw = extract_offline_transcripts("alameda_05fd2fe2-ce9f-48cf-8acc-c0a49d6a8067.mp3.transcript.json")
        else:
            chunk_transcripts_diarized, chunk_transcripts_raw = transcribe_live_chunks(api_key)
    else:
        chunk_transcripts_diarized, chunk_transcripts_raw = extract_offline_transcripts("alameda_05fd2fe2-ce9f-48cf-8acc-c0a49d6a8067.mp3.transcript.json")
        
    # Save transcripts of each chunk to disk
    os.makedirs("chunks", exist_ok=True)
    for i in range(1, 6):
        with open(os.path.join("chunks", f"chunk_{i}_transcript.txt"), "w", encoding="utf-8") as f:
            f.write(chunk_transcripts_diarized[i])
            
    # Concatenate all chunk transcripts for alignment
    concatenated_transcript = " ".join([chunk_transcripts_raw[i] for i in range(1, 6)])
    concatenated_norm = normalize(concatenated_transcript)
    local_words = concatenated_norm.split()
    
    # Step 2: Load ground truth data
    gt_path = "meetingbank_groundtruth.json"
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file '{gt_path}' not found. Please run download_meetingbank.py first.")
        return
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    # Target UIDs that fall inside the first 75 minutes of the meeting
    # 2019-6450 is from 7.56m to 17.40m
    # 2019-6945 is from 19.46m to 21.63m
    target_uids = ["AlamedaCC_06042019_2019-6450", "AlamedaCC_06042019_2019-6945"]
    
    evaluation_results = {
        "metadata": {
            "evaluation_type": "transcription",
            "source_audio": "alameda_05fd2fe2-ce9f-48cf-8acc-c0a49d6a8067.mp3",
            "number_of_chunks": 5,
            "chunk_size_minutes": 15
        },
        "chunks": {},
        "evaluations": {}
    }
    
    # Save transcripts per chunk to json
    for i in range(1, 6):
        evaluation_results["chunks"][f"chunk_{i}"] = {
            "file": os.path.join("chunks", f"chunk_{i}.mp3"),
            "transcript_file": os.path.join("chunks", f"chunk_{i}_transcript.txt"),
            "word_count": len(chunk_transcripts_diarized[i].split()),
            "text": chunk_transcripts_diarized[i]
        }
        
    print("\n--- Transcription (STT) Evaluation ---")
    total_wer = 0.0
    num_evals = 0
    aligned_segments = {}
    
    for uid in target_uids:
        if uid not in gt_data:
            print(f"Warning: UID '{uid}' not found in ground truth. Skipping.")
            continue
            
        gt_transcript = gt_data[uid]["transcript"]
        gt_norm = normalize(gt_transcript)
        gt_words = gt_norm.split()
        
        # Align
        matched_text, wer, start_idx, end_idx = find_best_alignment(local_words, gt_words)
        accuracy = max(0.0, 1.0 - wer)
        
        total_wer += wer
        num_evals += 1
        
        print(f"UID: {uid}")
        print(f"  Ground Truth Words: {len(gt_words)}")
        print(f"  Matched Segment Words: {len(matched_text.split())}")
        print(f"  Word Error Rate (WER): {wer:.2%}")
        print(f"  STT Accuracy: {accuracy:.2%}")
        print("-" * 40)
        
        evaluation_results["evaluations"][uid] = {
            "ground_truth_transcript": gt_transcript,
            "matched_transcript": matched_text,
            "word_error_rate": wer,
            "accuracy": accuracy
        }
        
        # Save aligned segments for evaluate_summary.py
        aligned_segments[uid] = {
            "matched_transcript": matched_text,
            "ground_truth_summary": gt_data[uid]["summary"]
        }
        
    if num_evals > 0:
        avg_wer = total_wer / num_evals
        avg_accuracy = 1.0 - avg_wer
        evaluation_results["metadata"]["average_word_error_rate"] = avg_wer
        evaluation_results["metadata"]["average_accuracy"] = avg_accuracy
        print(f"Average Word Error Rate (WER): {avg_wer:.2%}")
        print(f"Average STT Accuracy: {avg_accuracy:.2%}")
    else:
        print("No target UIDs were evaluated.")
        
    # Save results to transcription_evaluation.json
    with open("transcription_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print("\nTranscription evaluations saved to 'transcription_evaluation.json'")
    
    # Save aligned transcripts to aligned_transcripts.json
    with open("aligned_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(aligned_segments, f, indent=2)

if __name__ == "__main__":
    main()
