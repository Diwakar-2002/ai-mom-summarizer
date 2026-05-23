import os
import argparse
from dotenv import load_dotenv
import assemblyai as aai

try:
    import jiwer
except ImportError:
    jiwer = None

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
    parser.add_argument("--save", action="store_true", help="Save the transcript to a text file")
    parser.add_argument("--ground-truth", help="Path to a ground truth text file to calculate Word Error Rate (WER)")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (identify who is speaking)")
    args = parser.parse_args()

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
        if args.diarize and getattr(transcript, 'utterances', None):
            for utterance in transcript.utterances:
                print(f"Speaker {utterance.speaker}: {utterance.text}")
        else:
            print(transcript.text)
        print("------------------")

        # Calculate Accuracy / Confidence Score
        if getattr(transcript, 'words', None):
            avg_confidence = sum(word.confidence for word in transcript.words) / len(transcript.words)
            print(f"\nAverage Confidence Score: {avg_confidence:.2%} (How confident the AI is in this transcript)")
            
            # Show a few low-confidence words as examples
            low_confidence_words = [f"'{word.text}'({word.confidence:.2f})" for word in transcript.words if word.confidence < 0.8]
            if low_confidence_words:
                print(f"Words with <80% confidence: {', '.join(low_confidence_words[:10])}{'...' if len(low_confidence_words) > 10 else ''}")

        # Calculate Word Error Rate (WER) if ground truth is provided
        if args.ground_truth:
            if not os.path.exists(args.ground_truth):
                print(f"\nWarning: Ground truth file '{args.ground_truth}' not found. Skipping WER calculation.")
            elif not jiwer:
                print("\nWarning: 'jiwer' library is not installed. Please run 'pip install jiwer' to calculate WER.")
            else:
                try:
                    with open(args.ground_truth, "r", encoding="utf-8") as f:
                        ground_truth_text = f.read()
                    
                    # Normalize text for accurate WER calculation (lowercase, remove punctuation)
                    transformation = jiwer.Compose([
                        jiwer.ToLowerCase(),
                        jiwer.RemovePunctuation(),
                        jiwer.RemoveMultipleSpaces(),
                        jiwer.Strip(),
                        jiwer.ExpandCommonEnglishContractions()
                    ])
                    
                    gt_norm = transformation(ground_truth_text)
                    transcript_norm = transformation(transcript.text)
                    
                    error_rate = jiwer.wer(gt_norm, transcript_norm)
                    accuracy = 1.0 - error_rate
                    print(f"\n--- Accuracy Evaluation ---")
                    print(f"Word Error Rate (WER): {error_rate:.2%}")
                    print(f"Overall Accuracy: {accuracy:.2%}")
                    print("---------------------------")
                except Exception as e:
                    print(f"\nError calculating WER: {e}")

        if args.save:
            output_file = os.path.splitext(args.audio_file)[0] + "_transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                if args.diarize and getattr(transcript, 'utterances', None):
                    for utterance in transcript.utterances:
                        f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
                else:
                    f.write(transcript.text)
            print(f"\nTranscript saved to {output_file}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
