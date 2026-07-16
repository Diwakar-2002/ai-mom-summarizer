import os
import argparse
import json
from dotenv import load_dotenv
import assemblyai as aai
from summarize import generate_summary

def parse_mom_output(mom_output):
    if not mom_output:
        return "", ""
        
    action_headers = [
        "## ACTION ITEMS",
        "## WINNING IDEA / NEXT STEPS",
        "## NEXT STEPS & FOLLOW-UPS",
        "## IMMEDIATE NEXT STEPS",
        "## ACTIONABLE IMPROVEMENTS",
        "## ACTION ITEMS & INDIVIDUAL TASKS"
    ]
    
    for header in action_headers:
        if header in mom_output:
            parts = mom_output.split(header, 1)
            summary = parts[0].strip()
            action_items = f"{header}\n{parts[1].strip()}"
            return summary, action_items
            
    return mom_output, ""

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("Error: ASSEMBLYAI_API_KEY not found. Please set it in the .env file.")
        return

    # Parse arguments
    parser = argparse.ArgumentParser(description="Transcribe an audio file using AssemblyAI.")
    parser.add_argument("audio_file", help="Path to the audio file (e.g., .mp3)")
    parser.add_argument("--save", action="store_true", help="Save outputs (transcript, summary, and consolidated JSON)")
    parser.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization (enabled by default)")
    parser.add_argument("--summarize", action="store_true", help="Generate meeting summary locally using Ollama (phi3 model)")
    parser.add_argument(
        "--meeting-type", 
        default="general", 
        choices=["stand_up", "creative", "financial", "sales_pitch", "kickoff", "post_mortem", "one_on_one", "general"], 
        help="Type of meeting for dynamic prompt routing during summarization"
    )
    parser.add_argument(
        "--max-words", 
        type=int, 
        default=None, 
        help="Truncate transcript to maximum words for LLM summary input"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="phi3", 
        help="Candidate summarizer model (e.g., phi3, gemini, llama3, gemma)"
    )
    args = parser.parse_args()
    args.diarize = not args.no_diarize

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        return

    # Configure AssemblyAI
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=args.diarize
    )
    transcriber = aai.Transcriber(config=config)

    print(f"Uploading and transcribing '{args.audio_file}'...")
    try:
        transcript = transcriber.transcribe(args.audio_file)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription failed: {transcript.error}")
            return

        print("\n--- Transcript ---")
        full_transcript_text = ""
        if args.diarize and getattr(transcript, 'utterances', None):
            for utterance in transcript.utterances:
                line = f"Speaker {utterance.speaker}: {utterance.text}"
                print(line)
                full_transcript_text += line + "\n"
        else:
            print(transcript.text)
            full_transcript_text = transcript.text
        print("------------------")

        # Calculate Accuracy / Confidence Score
        avg_confidence = None
        if getattr(transcript, 'words', None):
            avg_confidence = sum(word.confidence for word in transcript.words) / len(transcript.words)
            print(f"\nAverage Confidence Score: {avg_confidence:.2%} (How confident the AI is in this transcript)")
            
            # Show a few low-confidence words as examples
            low_confidence_words = [f"'{word.text}'({word.confidence:.2f})" for word in transcript.words if word.confidence < 0.8]
            if low_confidence_words:
                print(f"Words with <80% confidence: {', '.join(low_confidence_words[:10])}{'...' if len(low_confidence_words) > 10 else ''}")

        mom_output = None
        if args.summarize:
            print(f"\nGenerating Meeting Summary using '{args.model}' with '{args.meeting_type}' prompt routing...")
            try:
                mom_output = generate_summary(
                    full_transcript_text, 
                    meeting_type=args.meeting_type,
                    max_words=args.max_words,
                    model_name=args.model
                )
                
                print("\n=== Minutes of Meeting (MoM) [Offline] ===")
                print(mom_output)
                print("==========================================")
                
            except Exception as e:
                print(f"\nError generating offline summary: {e}")

        if args.save:
            summary_part, action_items_part = parse_mom_output(mom_output)
            output_file = os.path.splitext(args.audio_file)[0] + "_output.txt"
            score_str = f"{avg_confidence:.2%}" if avg_confidence is not None else "N/A"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("==================================================\n")
                f.write(f"CONFIDENCE SCORE OF TRANSCRIPT: {score_str}\n")
                f.write("==================================================\n\n")
                
                f.write("==================================================\n")
                f.write("DIARIZED TRANSCRIPT\n")
                f.write("==================================================\n")
                f.write(f"{full_transcript_text.strip()}\n")
                
                f.write("==================================================\n")
                f.write("SUMMARY\n")
                f.write("==================================================\n")
                f.write(f"{summary_part.strip() if summary_part else 'No summary generated.'}\n\n")
                
                f.write("==================================================\n")
                f.write("ACTION ITEMS\n")
                f.write("==================================================\n")
                f.write(f"{action_items_part.strip() if action_items_part else 'No action items generated.'}\n\n")
                
            print(f"\nSaved all results to single output file: {output_file}")
            
        # Automatically save embeddings in ChromaDB
        meeting_id = os.path.splitext(os.path.basename(args.audio_file))[0]
        print(f"\nAutomatically saving meeting '{meeting_id}' to local ChromaDB database...")
        try:
            from rag_pipeline import store_meeting
            store_meeting(meeting_id, full_transcript_text, mom_output)
        except Exception as e:
            print(f"Failed to save to ChromaDB: {e}")
            
    except Exception as e:

        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
