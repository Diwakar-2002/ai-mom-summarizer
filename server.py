import os
import re
import sys
import json
import logging
import subprocess
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# Load env variables
load_dotenv()

# Import RAG and Summarize modules
from summarize import generate_summary
from rag_pipeline import store_meeting, ask_meeting_brain, init_db

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="AI MOM Summarizer & Evaluation Suite")

# Ensure upload directory exists
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper to clean item ID from UID (matches evaluate_single_audio.py logic)
def get_clean_item_id(uid: str) -> str:
    parts = uid.split('_')
    item_id = parts[-1]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', item_id)

class ChatRequest(BaseModel):
    query: str
    model_name: str = "phi3"
    meeting_id: Optional[str] = None

class EvaluateRequest(BaseModel):
    uid: str = "SeattleCityCouncil_06132016_Res 31669"
    model_name: str = "phi3"

@app.get("/api/status")
def get_status():
    """Checks the status of the local Ollama instance and returns available models."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [m["name"] for m in models_data.get("models", [])]
            return {
                "ollama_running": True,
                "models": models,
                "host": ollama_host
            }
    except Exception as e:
        logging.warning(f"Failed to connect to Ollama: {e}")
        
    return {
        "ollama_running": False,
        "models": [],
        "host": ollama_host,
        "note": "Make sure Ollama is serving (e.g. CUDA_VISIBLE_DEVICES='' ./bin/ollama serve)"
    }

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    meeting_type: str = Form("general"),
    model_name: str = Form("phi3")
):
    """Handles uploading an audio file, transcribes it via AssemblyAI, generates MOM summary, and stores in ChromaDB."""
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY not configured in env.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Transcribe via AssemblyAI
    logging.info(f"Transcribing {file_path} via AssemblyAI...")
    import assemblyai as aai
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=diarize
    )
    transcriber = aai.Transcriber(config=config)
    
    try:
        transcript_result = transcriber.transcribe(file_path)
        if transcript_result.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript_result.error}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AssemblyAI request failed: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    # Format transcript
    full_transcript_text = ""
    if diarize and getattr(transcript_result, 'utterances', None):
        for utterance in transcript_result.utterances:
            full_transcript_text += f"Speaker {utterance.speaker}: {utterance.text}\n"
    else:
        full_transcript_text = transcript_result.text

    # Generate Summary
    logging.info(f"Generating summary with type={meeting_type}, model={model_name}...")
    try:
        summary_text = generate_summary(full_transcript_text, meeting_type=meeting_type, model_name=model_name)
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        summary_text = f"Error generating summary: {e}"

    # Ingest into ChromaDB RAG brain
    meeting_id = os.path.splitext(file.filename)[0]
    # Clean meeting_id to be safe
    meeting_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', meeting_id)
    logging.info(f"Storing meeting {meeting_id} in vector DB...")
    try:
        store_meeting(meeting_id, full_transcript_text, summary_text)
    except Exception as e:
        logging.error(f"ChromaDB ingestion failed: {e}")

    return {
        "meeting_id": meeting_id,
        "transcript": full_transcript_text,
        "summary": summary_text
    }

@app.post("/api/chat")
def chat_with_brain(req: ChatRequest):
    """Queries ChromaDB vector store and answers user query about meetings."""
    try:
        answer = ask_meeting_brain(
            user_query=req.query,
            meeting_id=req.meeting_id,
            model_name=req.model_name
        )
        # Fetch matching metadata for UI display of sources
        collection = init_db()
        query_emb = requests.post(
            f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": req.query},
            timeout=30
        ).json().get("embedding")
        
        query_args = {"query_embeddings": [query_emb], "n_results": 3}
        if req.meeting_id:
            query_args["where"] = {"meeting_id": req.meeting_id}
            
        results = collection.query(**query_args)
        retrieved_metadata = results.get("metadatas", [[]])[0]
        retrieved_chunks = results.get("documents", [[]])[0]
        
        sources = [
            {"meeting_id": m.get("meeting_id"), "type": m.get("type"), "snippet": text[:200] + "..."}
            for m, text in zip(retrieved_metadata, retrieved_chunks)
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

@app.post("/api/evaluate")
def run_evaluation(req: EvaluateRequest):
    """Runs evaluate_single_audio.py for a given UID and model, parsing the generated report."""
    logging.info(f"Starting evaluation of UID: {req.uid} using model: {req.model_name}...")
    
    # Run the script as a subprocess
    cmd = [
        sys.executable,
        "evaluate_single_audio.py",
        "--uid", req.uid,
        "--model", req.model_name
    ]
    
    try:
        # Run process synchronously (with timeout of 10 minutes to support transcription + summary)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Evaluation pipeline script exited with code {result.returncode}",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Evaluation execution timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute evaluation script: {e}")

    # Read output reports
    clean_id = get_clean_item_id(req.uid)
    json_path = f"single_evaluation_report_{clean_id}.json"
    md_path = f"single_evaluation_report_{clean_id}.md"

    metrics = {}
    report_md = ""
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON metrics: {e}")
            
    if os.path.exists(md_path):
        try:
            with open(md_path, "r") as f:
                report_md = f.read()
        except Exception as e:
            logging.error(f"Failed to load Markdown report: {e}")

    return {
        "success": True,
        "metrics": metrics,
        "report_md": report_md,
        "stdout": result.stdout
    }

# Serve Static files (we will place index.html, style.css, app.js inside static/)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """Serves the main application page."""
    static_index = "./static/index.html"
    if os.path.exists(static_index):
        with open(static_index, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h2>Static frontend files not found yet. Please wait.</h2>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
