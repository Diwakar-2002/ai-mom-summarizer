import os
import re
import json
import argparse
from dotenv import load_dotenv
from datasets import load_dataset

def clean_and_normalize(text):
    # Remove speaker labels like "Speaker A: " or "SPEAKER B:"
    text = re.sub(r'(?i)speaker\s+[a-z]:', '', text)
    # Remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def read_local_transcript(file_path):
    if file_path.endswith(".json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "segments" in data:
                    texts = []
                    for seg in data["segments"]:
                        if "nbest" in seg and seg["nbest"]:
                            texts.append(seg["nbest"][0].get("text", ""))
                        elif "text" in seg:
                            texts.append(seg["text"])
                    return " ".join(texts)
                elif "text" in data:
                    return data["text"]
        except Exception as e:
            print(f"Error parsing JSON transcript {file_path}: {e}")
            return None
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text transcript {file_path}: {e}")
            return None

def find_transcript_for_audio(audio_path):
    dir_name = os.path.dirname(audio_path) or "."
    base_name = os.path.basename(audio_path)
    root, ext = os.path.splitext(base_name)
    
    candidates = [
        os.path.join(dir_name, f"{root}_transcript.txt"),
        os.path.join(dir_name, f"{root}.txt"),
        os.path.join(dir_name, f"{base_name}_transcript.txt"),
        os.path.join(dir_name, f"{base_name}.txt"),
        os.path.join(dir_name, f"{base_name}.transcript.json"),
        os.path.join(dir_name, f"{root}.transcript.json"),
    ]
    
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None

def find_meeting_prefix_by_matching(local_transcript_text, ds):
    print("Searching for matching meeting in Hugging Face dataset...")
    local_norm = clean_and_normalize(local_transcript_text)
    local_words = local_norm.split()
    
    if len(local_words) < 50:
        print("Local transcript is too short to perform matching.")
        return None
        
    # Sample 5 passages of 35 words from different parts of the transcript
    passages = []
    step = len(local_words) // 6
    for i in range(1, 6):
        start = i * step
        passage = " ".join(local_words[start:start+35])
        if len(passage.split()) >= 15:
            passages.append(passage)
            
    if not passages:
        print("Could not extract enough valid passages for matching.")
        return None
        
    best_match_prefix = None
    max_matches = 0
    
    for split in ds.keys():
        for item in ds[split]:
            gt_transcript = clean_and_normalize(item.get("transcript", ""))
            matches = sum(1 for p in passages if p in gt_transcript)
            if matches > max_matches:
                max_matches = matches
                uid = item.get("uid", "")
                if "_" in uid:
                    parts = uid.split("_")
                    if len(parts) >= 2:
                        best_match_prefix = "_".join(parts[:2])
            if max_matches >= 3:
                print(f"Strong match found for prefix: {best_match_prefix}")
                return best_match_prefix
                
    if best_match_prefix:
        print(f"Best match prefix found: {best_match_prefix} (confidence: {max_matches}/5)")
    return best_match_prefix

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Download ground-truth transcripts and summaries from Hugging Face MeetingBank.")
    parser.add_argument("--uid", nargs="+", help="Specific UID(s) to download")
    parser.add_argument("--meeting", nargs="+", help="Specific meeting prefix(es) to download (e.g., BostonCC_01032022)")
    parser.add_argument("--auto", action="store_true", help="Force auto-detect from local audio files and transcripts")
    
    args = parser.parse_args()
    
    target_uids = set(args.uid or [])
    target_prefixes = set(args.meeting or [])
    
    # Fallback to auto-detection if no manual options are specified
    if not args.uid and not args.meeting:
        print("No specific UIDs or meeting prefixes specified. Scanning directory for local audio files...")
        audio_extensions = (".mp3", ".wav", ".m4a", ".flac")
        audio_files = [f for f in os.listdir(".") if f.lower().endswith(audio_extensions)]
        
        if audio_files:
            print(f"Found local audio files: {audio_files}")
            print("Loading MeetingBank dataset from Hugging Face for mapping...")
            ds = load_dataset("huuuyeah/meetingbank")
            
            for audio in audio_files:
                transcript_path = find_transcript_for_audio(audio)
                if not transcript_path:
                    print(f"No transcript file found for {audio}. Skipping.")
                    continue
                
                print(f"Using transcript file: {transcript_path} for auto-detection...")
                local_text = read_local_transcript(transcript_path)
                if not local_text:
                    continue
                    
                prefix = find_meeting_prefix_by_matching(local_text, ds)
                if prefix:
                    print(f"Mapped {audio} -> Meeting prefix: {prefix}")
                    target_prefixes.add(prefix)
                else:
                    print(f"Could not map {audio} to any meeting in the dataset.")
        else:
            print("No local audio files found in workspace.")
            
    # Default fallback: if still nothing to download, default to the original Alameda benchmark UIDs
    if not target_uids and not target_prefixes:
        print("Defaulting to the original Alameda benchmark UIDs...")
        target_uids = {
            "AlamedaCC_06042019_2019-6450",
            "AlamedaCC_06042019_2019-6895",
            "AlamedaCC_06042019_2019-6897",
            "AlamedaCC_06042019_2019-6901",
            "AlamedaCC_06042019_2019-6902",
            "AlamedaCC_06042019_2019-6917",
            "AlamedaCC_06042019_2019-6945",
            "AlamedaCC_06042019_2019-6948"
        }
        
    # Start download
    ground_truth = {}
    gt_path = "meetingbank_groundtruth.json"
    if os.path.exists(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            print(f"Loaded {len(ground_truth)} existing ground-truth records from {gt_path}")
        except Exception as e:
            print(f"Could not load existing ground-truth JSON: {e}")
            
    print(f"Target UIDs to download: {target_uids}")
    print(f"Target Meeting prefixes to download: {target_prefixes}")
    
    # Load dataset if not loaded yet
    if 'ds' not in locals():
        print("Loading MeetingBank dataset from Hugging Face...")
        try:
            ds = load_dataset("huuuyeah/meetingbank")
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            return
            
    new_downloads_count = 0
    for split in ds.keys():
        print(f"Searching in split: {split}...")
        for item in ds[split]:
            uid = item.get("uid")
            if not uid:
                continue
                
            match = False
            if uid in target_uids:
                match = True
            else:
                for prefix in target_prefixes:
                    if uid.startswith(prefix):
                        match = True
                        break
                        
            if match:
                if uid not in ground_truth:
                    new_downloads_count += 1
                    print(f"Found new match: {uid} in split {split}")
                ground_truth[uid] = {
                    "split": split,
                    "summary": item.get("summary", ""),
                    "transcript": item.get("transcript", "")
                }
                
    print(f"Downloaded {new_downloads_count} new ground-truth records.")
    
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth data to {gt_path}")

if __name__ == "__main__":
    main()
