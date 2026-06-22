import os
import argparse
import requests
import json

# Prompts Library
PROMPTS = {
    "stand_up": """You are a software project manager. Read the following daily stand-up transcript.
Extract exactly three things for each person who spoke:
1. What they did yesterday.
2. What they are doing today.
3. Any BLOCKERS or issues preventing them from working.

OUTPUT FORMAT:
## [Person's Name]
- Done: [Task]
- Doing: [Task]
- Blockers: [Issue or "None"]""",

    "creative": """You are a creative director. Read the following brainstorming transcript.
Do not look for financials. Focus entirely on the ideas discussed.

OUTPUT FORMAT:
## CORE GOAL OF BRAINSTORM
[1 sentence summary]

## IDEAS PROPOSED
- [Idea 1]: [Brief description and who proposed it]
- [Idea 2]: [Brief description and who proposed it]

## WINNING IDEA / NEXT STEPS
[Which idea did the team seem to agree on the most, and what is the next step?]""",

    "financial": """You are an expert corporate secretary. Read the following meeting transcript and generate a strict Minutes of Meeting (MOM) document.

RULES:
1. Do NOT analyze the text. Do NOT use phrases like "in the transcript" or "Speaker B said".
2. Ignore procedural votes, moving, seconding, and adjourning. Only extract business decisions.
3. Extract all exact dollar amounts, deadlines, and project names.

OUTPUT FORMAT (Do not output anything else):
## SUMMARY
[Write a 2-sentence summary of the main decisions made.]

## KEY DECISIONS & FINANCIALS
- [Bullet points of specific projects, dollar amounts, or numbers mentioned]

## ACTION ITEMS
- [Action] - [Deadline if mentioned]""",

    "sales_pitch": """You are an expert sales strategist. Read the following sales pitch transcript.
Analyze the interaction and extract the following details.

OUTPUT FORMAT:
## CLIENT NEEDS & PAIN POINTS
- [What problems is the client trying to solve?]

## VALUE PROPOSITION & PROPOSED SOLUTION
- [What solution did the presenter offer, and what are its key benefits?]

## OBJECTIONS & CONCERNS
- [What concerns or objections did the client raise, and how were they addressed?]

## NEXT STEPS & FOLLOW-UPS
- [What are the agreed next steps, follow-up meetings, or deliverables?]""",

    "kickoff": """You are a senior project manager leading a kickoff meeting. Read the following kickoff meeting transcript.
Extract the project launch details.

OUTPUT FORMAT:
## PROJECT OBJECTIVES & SCOPE
- [What is the main goal of the project and what are the boundaries of the scope?]

## ROLES & RESPONSIBILITIES
- [Who is on the team and what are their specific assignments or roles?]

## MILESTONES & TIMELINE
- [What are the key deadlines, deliverables, and timeline milestones?]

## IMMEDIATE NEXT STEPS
- [What must the team work on immediately in the first phase?]""",

    "post_mortem": """You are an agile coach conducting a post-mortem / retrospective meeting. Read the following transcript.
Analyze the project outcomes discussed.

OUTPUT FORMAT:
## WHAT WENT WELL
- [Successful aspects, achievements, and positive outcomes]

## WHAT WENT WRONG & CHALLENGES
- [Bottlenecks, issues, and failures encountered]

## KEY LESSONS LEARNED
- [Insights gained for future projects]

## ACTIONABLE IMPROVEMENTS
- [Specific process improvements or tasks for next time]""",

    "one_on_one": """You are an empathetic manager reviewing a 1-on-1 meeting transcript.
Extract the key personal and professional insights from the conversation.

OUTPUT FORMAT:
## CAREER DEVELOPMENT & GOALS
- [Discussion on goals, aspirations, and career growth]

## FEEDBACK & SUPPORT NEEDED
- [Feedback shared and support requested from the manager or team]

## ACTION ITEMS & INDIVIDUAL TASKS
- [Agreed action items for the employee before the next 1-on-1]""",

    "general": """You are a corporate assistant. Read the following meeting transcript.
Summarize the meeting and list action items.

OUTPUT FORMAT:
## SUMMARY
- [A concise summary of the overall meeting discussion]

## ACTION ITEMS
- [Action item] - [Assigned to / Deadline if mentioned]"""
}

def generate_summary(transcript_text, meeting_type="general", skip_eval=False, max_words=None):
    """
    Generate meeting summary locally using Ollama (phi3 model) with dynamic routing.
    """
    if max_words:
        words = transcript_text.split()
        if len(words) > max_words:
            transcript_text = " ".join(words[:max_words]) + "\n[Transcript truncated for length...]"
            
    active_prompt = PROMPTS.get(meeting_type, PROMPTS["general"])
    prompt = f"{active_prompt}\n\nTranscript:\n{transcript_text}"
    
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "top_p": 0.9,
            "num_predict": 300
        }
    }
    
    # Send request to Ollama with a timeout of 600 seconds
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
    response.raise_for_status()
    mom_output = response.json().get("response", "")
    
    if skip_eval:
        return mom_output

    # Evaluate summary accuracy
    eval_prompt = (
        "You are an AI auditor. Please evaluate the following summary of a meeting transcript. "
        "Rate the summary's accuracy and coverage of the transcript on a scale of 1-10. "
        "Just provide a single score like '9/10' and a 1 sentence reason.\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Summary to evaluate:\n{mom_output}"
    )
    eval_payload = {
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
    
    eval_response = requests.post("http://localhost:11434/api/generate", json=eval_payload, timeout=600)
    eval_response.raise_for_status()
    eval_output = eval_response.json().get("response", "").strip()
    
    full_output = f"{mom_output}\n\n=== AI Self-Evaluation ===\n{eval_output}\n=========================="
    return full_output



def main():
    parser = argparse.ArgumentParser(description="Summarize a meeting transcript locally using Ollama.")
    parser.add_argument("transcript_file", help="Path to the transcript text file")
    parser.add_argument(
        "--type", 
        choices=list(PROMPTS.keys()), 
        default="general", 
        help="Type of meeting for dynamic prompt routing"
    )
    parser.add_argument("--save", action="store_true", help="Save the summary to a file")
    args = parser.parse_args()

    if not os.path.exists(args.transcript_file):
        print(f"Error: Transcript file '{args.transcript_file}' not found.")
        return

    try:
        with open(args.transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        # Remove any metadata metrics blocks if they exist in the file
        if "--- Accuracy Metrics ---" in transcript_text:
            transcript_text = transcript_text.split("--- Accuracy Metrics ---")[0].strip()

        print(f"Generating summary using '{args.type}' prompt routing...")
        summary = generate_summary(transcript_text, args.type)
        
        print("\n=== Meeting Summary ===")
        print(summary)
        print("=======================")

        if args.save:
            base, ext = os.path.splitext(args.transcript_file)
            if base.endswith("_transcript"):
                out_name = base[:-11] + "_offline_mom.txt"
            else:
                out_name = base + "_offline_mom.txt"

            with open(out_name, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"\nSummary saved to: {out_name}")

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API. Make sure Ollama is running. Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
