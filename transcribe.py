import os
import argparse
from dotenv import load_dotenv
import assemblyai as aai
import requests
import json
try:
    import jiwer
except ImportError:
    jiwer = None

from summarize import generate_summary

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from evaluate_summary import get_llm_judge_score
except ImportError:
    get_llm_judge_score = None

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
    parser.add_argument("--ground-truth-summary", help="Path to a ground truth summary text file to evaluate summarization quality")
    parser.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization (enabled by default)")
    parser.add_argument("--summarize", action="store_true", help="Generate meeting summary locally using Ollama (phi3 model)")
    parser.add_argument(
        "--meeting-type", 
        default="general", 
        choices=["stand_up", "creative", "financial", "sales_pitch", "kickoff", "post_mortem", "one_on_one", "general"], 
        help="Type of meeting for dynamic prompt routing during summarization"
    )
    parser.add_argument(
        "--skip-eval", 
        action="store_true", 
        default=False, 
        help="Skip AI self-evaluation step for faster summary generation"
    )
    parser.add_argument(
        "--max-words", 
        type=int, 
        default=None, 
        help="Truncate transcript to maximum words for LLM summary input"
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

        # Calculate Word Error Rate (WER) if ground truth is provided
        accuracy = None
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
        if args.summarize:
            print(f"\nGenerating Meeting Summary locally using Ollama (phi3) with '{args.meeting_type}' prompt routing...")
            try:
                mom_output = generate_summary(
                    full_transcript_text, 
                    meeting_type=args.meeting_type,
                    skip_eval=args.skip_eval,
                    max_words=args.max_words
                )
                
                # Strip AI self-evaluation tags if --skip-eval was used and returned a clean summary
                mom_clean = mom_output.split("\n\n=== AI Self-Evaluation ===")[0].strip()
                
                print("\n=== Minutes of Meeting (MoM) [Offline] ===")
                print(mom_clean)
                print("==========================================")
                
                # Use clean MoM output for saving and print
                mom_output = mom_clean
                
                # Calculate summary accuracy metrics if ground truth summary is provided
                summary_metrics = None
                if args.ground_truth_summary:
                    if not os.path.exists(args.ground_truth_summary):
                        print(f"\nWarning: Ground truth summary file '{args.ground_truth_summary}' not found. Skipping evaluation.")
                    elif not rouge_scorer:
                        print("\nWarning: 'rouge-score' library is not installed. Skipping ROUGE calculation.")
                    else:
                        try:
                            with open(args.ground_truth_summary, "r", encoding="utf-8") as f:
                                gt_summary_text = f.read().strip()
                            
                            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                            scores = scorer.score(gt_summary_text, mom_clean)
                            
                            r1 = scores['rouge1'].fmeasure
                            r2 = scores['rouge2'].fmeasure
                            rl = scores['rougeL'].fmeasure
                            
                            print(f"\n--- Summarization Evaluation (ROUGE) ---")
                            print(f"ROUGE-1 F1: {r1:.2%}")
                            print(f"ROUGE-2 F1: {r2:.2%}")
                            print(f"ROUGE-L F1: {rl:.2%}")
                            
                            judge_score = None
                            judge_feedback = None
                            if get_llm_judge_score:
                                print("Evaluating summary using LLM-as-a-judge...")
                                judge_score, judge_feedback = get_llm_judge_score(gt_summary_text, mom_clean)
                                print(f"LLM Judge Score: {judge_score}/10")
                                print(f"Feedback: {judge_feedback}")
                            
                            print("----------------------------------------")
                            
                            summary_metrics = {
                                "rouge1": r1,
                                "rouge2": r2,
                                "rougeL": rl,
                                "judge_score": judge_score,
                                "judge_feedback": judge_feedback
                            }
                        except Exception as e:
                            print(f"\nError calculating summary metrics: {e}")
                
                if args.save:
                    mom_file = os.path.splitext(args.audio_file)[0] + "_offline_mom.txt"
                    with open(mom_file, "w", encoding="utf-8") as f:
                        f.write(mom_output)
                        if summary_metrics:
                            f.write("\n\n--- Summarization Evaluation ---\n")
                            f.write(f"ROUGE-1 F1: {summary_metrics['rouge1']:.2%}\n")
                            f.write(f"ROUGE-2 F1: {summary_metrics['rouge2']:.2%}\n")
                            f.write(f"ROUGE-L F1: {summary_metrics['rougeL']:.2%}\n")
                            if summary_metrics['judge_score'] is not None:
                                f.write(f"LLM Judge Score: {summary_metrics['judge_score']}/10\n")
                                f.write(f"LLM Judge Feedback: {summary_metrics['judge_feedback']}\n")
                            f.write("--------------------------------\n")
                    print(f"\nOffline Minutes of Meeting saved to {mom_file}")
            except Exception as e:
                print(f"\nError generating offline summary: {e}")

        if args.save:
            output_file = os.path.splitext(args.audio_file)[0] + "_transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_transcript_text.strip())
                f.write("\n\n--- Accuracy Metrics ---\n")
                if avg_confidence is not None:
                    f.write(f"Average Confidence Score: {avg_confidence:.2%}\n")
                if accuracy is not None:
                    f.write(f"Overall Accuracy (1-WER): {accuracy:.2%}\n")
                f.write("------------------------\n")
            print(f"\nTranscript saved to {output_file}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
