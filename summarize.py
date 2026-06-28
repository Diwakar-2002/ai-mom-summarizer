import os
import argparse
import requests
import json

# Prompts Library

# Shared anti-hallucination grounding rules, injected into every prompt below.
# Rationale: small/local models (e.g. Phi-3) tend to "overcorrect" noisy STT text —
# instead of fixing only what context clearly supports, they invent plausible-sounding
# but wrong names/numbers/words. These rules gate correction behind confidence and
# forbid fabrication outright.
 
GROUNDING_RULES = """
ANTI-HALLUCINATION RULES (apply strictly):
- Base every statement ONLY on what is explicitly said in the transcript. Do not infer, assume, or add information not directly stated.
- This is a raw Speech-to-Text (STT) transcript and will contain phonetic errors, mis-transcribed names, and garbled numbers (e.g. "cancel authorized borrowing" -> "council authorized borrowing", "arm of Springfield" -> "RM of Springfield").
- Correct an STT error ONLY when surrounding context makes the correct word/number/name UNAMBIGUOUS. If you are not highly confident, keep the original transcribed text and append "[unclear]" rather than guessing a replacement.
- NEVER invent a number, amount, date, or name that has no basis in the transcript. If the same entity is mentioned multiple times with conflicting transcriptions (e.g. "Ethan weed" later "Ethan Wiebe"), pick the clearest/most complete spelling and use it consistently — do not create a third, new variant.
- For numbers/amounts: if two mentions of the same figure disagree, use the one with more contextual support (e.g. repeated, or confirmed by a follow-up statement) and do not average, round, or combine them.
- If a required field has no information in the transcript, output "None". Do not fabricate a plausible-sounding answer to fill a field.
- Do not pad descriptions with generic detail the speaker never said, even if it would make the summary read more "complete".
"""
 
PROMPTS = {
    "stand_up": GROUNDING_RULES + """
You are a software project manager. Read the following daily stand-up transcript.
Extract exactly three things for each person who spoke:
1. What they did yesterday.
2. What they are doing today.
3. Any BLOCKERS or issues preventing them from working.
 
Use the person's name exactly as it is clearly spoken/confirmed in the transcript (apply the
grounding rules above if their name is garbled by STT). If a person's name is genuinely
unrecoverable, label them "Speaker [N]" rather than guessing a name.
 
OUTPUT FORMAT:
## [Person's Name]
- Done: [Task]
- Doing: [Task]
- Blockers: [Issue or "None"]
""",
 
    "creative": GROUNDING_RULES + """
You are a creative director. Read the following brainstorming transcript.
Do not look for financials. Focus entirely on the ideas discussed.
 
Attribute each idea to a specific person only if the transcript clearly indicates who proposed
it. If attribution is unclear, write "Unclear who proposed this" instead of guessing a name.
 
OUTPUT FORMAT:
## CORE GOAL OF BRAINSTORM
[1 sentence summary]
 
## IDEAS PROPOSED
- [Idea 1]: [Brief description and who proposed it]
- [Idea 2]: [Brief description and who proposed it]
 
## WINNING IDEA / NEXT STEPS
[Which idea did the team seem to agree on the most, and what is the next step?]
""",
 
    "financial": GROUNDING_RULES + """
You are an expert corporate secretary. Read the following meeting transcript and generate a
strict Minutes of Meeting (MOM) document.
 
This transcript involves council/board proceedings where exact dollar amounts, bylaw numbers,
and resolution outcomes carry legal/financial weight — accuracy here matters more than fluency.
Apply the grounding rules above with extra strictness to all numbers and proper nouns:
- For dollar amounts, only state a figure if it is stated clearly or confirmed by repetition/
  context elsewhere in the transcript. If a figure is ambiguous (e.g. "152" vs "150 too"),
  report the version supported by surrounding context and do not silently pick one without basis.
- For names of municipalities, districts, or organizations (e.g. "RM of Springfield"), only
  normalize capitalization/wording if the full term is unambiguous from context.
 
RULES:
1. In council or board meetings, decisions are made through resolutions and votes (e.g., motions
   carried or approved). Extract these as official business decisions.
2. Focus on extracting decisions, approvals (such as bylaw readings, financial statements, rates,
   or funding requests), exact dollar amounts, and project names.
3. Extract any action items, requests, or assignments (e.g., requesting printed copies, uploading
   documents to the website, preparing a new bylaw version) even if no explicit deadline is
   mentioned.
 
OUTPUT FORMAT (Do not output anything else):
## SUMMARY
[Write a 2-sentence summary of the main decisions made.]
 
## KEY DECISIONS & FINANCIALS
- [Bullet points of specific projects, approved bylaws, funding approvals, or numbers mentioned]
 
## ACTION ITEMS
- [Action item / request] - [Assignee or "None"] - [Deadline or "None"]
""",
 
    "sales_pitch": GROUNDING_RULES + """
You are an expert sales strategist. Read the following sales pitch transcript.
Analyze the interaction and extract the following details.
 
Only attribute a pain point, objection, or commitment to the client if they (or someone
representing them) actually said it. Do not infer a client's underlying need beyond what
was explicitly expressed.
 
OUTPUT FORMAT:
## CLIENT NEEDS & PAIN POINTS
- [What problems is the client trying to solve?]
 
## VALUE PROPOSITION & PROPOSED SOLUTION
- [What solution did the presenter offer, and what are its key benefits?]
 
## OBJECTIONS & CONCERNS
- [What concerns or objections did the client raise, and how were they addressed?]
 
## NEXT STEPS & FOLLOW-UPS
- [What are the agreed next steps, follow-up meetings, or deliverables?]
""",
 
    "kickoff": GROUNDING_RULES + """
You are a senior project manager leading a kickoff meeting. Read the following kickoff meeting
transcript. Extract the project launch details.
 
Only list a person as having a role/responsibility if the transcript explicitly assigns it to
them. Only list a date/milestone if a specific timeframe was actually stated — do not invent a
deadline because a task "should probably" have one.
 
OUTPUT FORMAT:
## PROJECT OBJECTIVES & SCOPE
- [What is the main goal of the project and what are the boundaries of the scope?]
 
## ROLES & RESPONSIBILITIES
- [Who is on the team and what are their specific assignments or roles?]
 
## MILESTONES & TIMELINE
- [What are the key deadlines, deliverables, and timeline milestones?]
 
## IMMEDIATE NEXT STEPS
- [What must the team work on immediately in the first phase?]
""",
 
    "post_mortem": GROUNDING_RULES + """
You are an agile coach conducting a post-mortem / retrospective meeting. Read the following
transcript. Analyze the project outcomes discussed.
 
Only list a "lesson learned" or "improvement" if the team explicitly discussed it as such — do
not generate generic agile-coaching advice that wasn't actually said in the meeting.
 
OUTPUT FORMAT:
## WHAT WENT WELL
- [Successful aspects, achievements, and positive outcomes]
 
## WHAT WENT WRONG & CHALLENGES
- [Bottlenecks, issues, and failures encountered]
 
## KEY LESSONS LEARNED
- [Insights gained for future projects]
 
## ACTIONABLE IMPROVEMENTS
- [Specific process improvements or tasks for next time]
""",
 
    "one_on_one": GROUNDING_RULES + """
You are an empathetic manager reviewing a 1-on-1 meeting transcript.
Extract the key personal and professional insights from the conversation.
 
This is sensitive personal content — be especially careful not to embellish feelings, opinions,
or career goals beyond what the employee actually expressed. Do not characterize tone (e.g.
"frustrated", "excited") unless the transcript supports it.
 
OUTPUT FORMAT:
## CAREER DEVELOPMENT & GOALS
- [Discussion on goals, aspirations, and career growth]
 
## FEEDBACK & SUPPORT NEEDED
- [Feedback shared and support requested from the manager or team]
 
## ACTION ITEMS & INDIVIDUAL TASKS
- [Agreed action items for the employee before the next 1-on-1]
""",
 
    "general": GROUNDING_RULES + """
You are a corporate assistant. Read the following meeting transcript.
Summarize the meeting and list action items.
 
OUTPUT FORMAT:
## SUMMARY
- [A concise summary of the overall meeting discussion]
 
## ACTION ITEMS
- [Action item] - [Assigned to / Deadline if mentioned]
""",
}
def generate_summary(transcript_text, meeting_type="general", max_words=None, model_name="phi3"):
    """
    Generate meeting summary locally using Ollama or via Gemini API.
    """
    if max_words:
        words = transcript_text.split()
        if len(words) > max_words:
            transcript_text = " ".join(words[:max_words]) + "\n[Transcript truncated for length...]"
            
    active_prompt = PROMPTS.get(meeting_type, PROMPTS["general"])
    prompt = f"{active_prompt}\n\nTranscript:\n{transcript_text}"
    
    if model_name == "gemini":
        # Call Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found in .env or environment.")
            
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        try:
            return res_json['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response structure from Gemini API: {res_json}") from e
    else:
        # Local Ollama
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 16384,
                "top_p": 0.9,
                "num_predict": 1000
            }
        }
        
        import requests
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=1800)
        response.raise_for_status()
        mom_output = response.json().get("response", "")
        return mom_output

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
    parser.add_argument("--model", default="phi3", help="Model name for summary generation (e.g. phi3, gemini)")
    args = parser.parse_args()

    if not os.path.exists(args.transcript_file):
        print(f"Error: Transcript file '{args.transcript_file}' not found.")
        return

    try:
        from dotenv import load_dotenv
        load_dotenv()

        with open(args.transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        if "--- Accuracy Metrics ---" in transcript_text:
            transcript_text = transcript_text.split("--- Accuracy Metrics ---")[0].strip()

        print(f"Generating summary using '{args.type}' prompt routing with model '{args.model}'...")
        summary = generate_summary(transcript_text, args.type, model_name=args.model)
        
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
