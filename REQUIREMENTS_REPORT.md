# AI MOM Summarizer - Dependencies & Requirements Reference

This document provides a comprehensive overview of every library listed in `requirements.txt`, explaining what it is and exactly where and how it is used in the **AI MOM Summarizer & Evaluation Suite** codebase. It serves as a study guide for code walkthroughs and project viva sessions.

---

## 🎙️ Speech-to-Text & Audio Processing

### 1. `assemblyai`
*   **What it is:** The official Python SDK for AssemblyAI, a high-accuracy, cloud-hosted Speech-to-Text API.
*   **Where & How it is used:**
    *   [transcribe.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/transcribe.py#L71-L85): Authenticates using the configured API key and calls the transcriber. It runs transcriptions with **Speaker Diarization** enabled to distinguish different speakers (e.g., "Speaker A", "Speaker B").
    *   [server.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/server.py#L91-L106): Powers the `/api/transcribe` endpoint, transcribing uploaded audio files on-the-fly and outputting speaker-segmented text.

### 2. `soundfile`
*   **What it is:** An audio library based on libsndfile, designed to read and write sound files (WAV, FLAC, etc.) in Python as NumPy arrays.
*   **Where & How it is used:**
    *   [evaluate_single_audio.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/evaluate_single_audio.py#L53): Used to inspect the downloaded audio files, determine sample rates and lengths, and extract precise clips/segments locally before sending them to the transcription API.

---

## 🤖 LLM, Embeddings & Vector Database (RAG)

### 3. `chromadb`
*   **What it is:** An open-source, lightweight vector database designed to store, manage, and query embeddings for AI/RAG applications.
*   **Where & How it is used:**
    *   [rag_pipeline.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/rag_pipeline.py): Initializes the persistent storage collection (`meeting_memory`) under the `meeting_db/` directory. It is used to upsert vector embeddings of transcript chunks and meeting summaries.
    *   [chat.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/chat.py): Queries the ChromaDB collection to retrieve the top 3 most relevant segments of transcripts for a user's question, feeding them as context to the chatbot's prompt.

### 4. `requests`
*   **What it is:** The standard HTTP library for Python, used to send synchronous HTTP requests.
*   **Where & How it is used:**
    *   [rag_pipeline.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/rag_pipeline.py#L34-L54): Connects to the local Ollama API endpoint `/api/embeddings` to generate vector representations of chunks and queries.
    *   [summarize.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/summarize.py#L246-L250): Connects to `/api/generate` to send prompts and get summaries from local LLMs.
    *   [server.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/server.py): Pings Ollama at `/api/tags` to check server status and list installed local models.

---

## 📐 Evaluation Metrics & Machine Learning

### 5. `bert-score`
*   **What it is:** An NLP evaluation metric that computes semantic similarity between two sentences using contextual token embeddings from models like RoBERTa or BERT.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py#L52-L57): Measures how well the generated candidate summary captures the meaning of the reference summary. It is superior to exact n-gram matching because it rewards correct paraphrasing.

### 6. `sentence-transformers`
*   **What it is:** A framework for generating dense vector representations (embeddings) for sentences and paragraphs.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py#L44-L50): Loads the `all-MiniLM-L6-v2` embedding model to calculate the **Cosine Similarity** between the sentence embeddings of candidate summaries and reference summaries. It also aligns and matches extracted action-item triples.

### 7. `summac`
*   **What it is:** An advanced summarization consistency detection framework. It parses text into sentences and evaluates entailment/contradiction scores using Natural Language Inference (NLI) models.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py#L59-L60): Evaluates the **Factual Consistency** of the generated summaries against the source transcripts to identify hallucinations.

### 8. `rouge-score`
*   **What it is:** Computes ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation), which measure n-gram lexical overlap (ROUGE-1, ROUGE-2, ROUGE-L) between candidate and reference summaries.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py#L8): Evaluates the lexical precision, recall, and F-measure of the generated summaries.

### 9. `evaluate`
*   **What it is:** A Hugging Face library designed to compute metrics, comparisons, and measurements for models and datasets in a standardized way.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py): Facilitates clean loading of traditional translation/summarization metrics like BLEU and METEOR.

### 10. `jiwer`
*   **What it is:** A Python library to compute Word Error Rate (WER) and Character Error Rate (CER) between reference and hypothesis text sequences.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py#L315-L355): Evaluates transcription accuracy by comparing AssemblyAI's generated transcripts against human-annotated MeetingBank transcripts.

### 11. `nltk`
*   **What it is:** Natural Language Toolkit; a classic library for text preprocessing, tokenization, and linguistic parsing.
*   **Where & How it is used:**
    *   [metrics.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/metrics.py#L5-L7): Splits the generated text summaries and transcripts into individual sentences (`sent_tokenize`) for Lead-N baseline generation, sentence-level matching, and metric alignment.

### 12. `transformers`
*   **What it is:** Hugging Face’s model library, providing thousands of pre-trained models for Natural Language Processing.
*   **Where & How it is used:**
    *   Imported implicitly by evaluation tools (`bert-score`, `summac`, and `sentence-transformers`) and used directly to build an NLI pipeline (`facebook/bart-large-mnli`) as a CPU-friendly fallback consistency evaluator.

### 13. `torch` (PyTorch)
*   **What it is:** An open-source machine learning library used for tensor computation and deep learning.
*   **Where & How it is used:**
    *   Serves as the underlying execution engine running all the local evaluation neural models (Transformers, BERTScore, SentenceTransformers) on either CPU or GPU.

---

## 🌐 Web Server & APIs

### 14. `fastapi`
*   **What it is:** A modern, high-performance web framework for building RESTful APIs in Python using standard type hints.
*   **Where & How it is used:**
    *   [server.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/server.py#L9): Configures routing, CORS, endpoints (`/api/transcribe`, `/api/chat`, `/api/evaluate`), background tasks, and error handling for the application.

### 15. `uvicorn`
*   **What it is:** An ASGI (Asynchronous Server Gateway Interface) web server implementation for Python.
*   **Where & How it is used:**
    *   [server.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/server.py#L254-L255): Hosts the FastAPI web backend on `http://localhost:8000`.

### 16. `python-multipart`
*   **What it is:** A streaming multipart form parser for Python web servers.
*   **Where & How it is used:**
    *   [server.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/server.py): Allows the FastAPI backend to parse incoming audio files and metadata fields sent from forms in the frontend user interface.

---

## 🛠️ Data Handling & Utilities

### 17. `python-dotenv`
*   **What it is:** A library that parses `.env` files and loads configurations into environmental variables.
*   **Where & How it is used:**
    *   Utilized across all files to load the `ASSEMBLYAI_API_KEY` and optional `GEMINI_API_KEY` into memory dynamically without hardcoding secret keys.

### 18. `datasets`
*   **What it is:** A Hugging Face library designed to download, query, and stream public research datasets.
*   **Where & How it is used:**
    *   [evaluate_single_audio.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/evaluate_single_audio.py#L297-L312): Streams/downloads the `huuuyeah/meetingbank` dataset test split to match the correct ground truth summaries.

### 19. `huggingface_hub`
*   **What it is:** A library allowing users to interact with the Hugging Face Hub (downloading models, configurations, and raw files).
*   **Where & How it is used:**
    *   [evaluate_single_audio.py](file:///home/Diwakar/Desktop/ai-mom-summarizer/evaluate_single_audio.py#L52): Fetches individual supplementary mapping files (such as `audio_zip_mapping.json`) stored on Hugging Face repositories.

### 20. `pandas`
*   **What it is:** A powerful data manipulation and analysis library that provides the `DataFrame` structure.
*   **Where & How it is used:**
    *   Handles tabular metrics aggregation in the evaluation pipeline and exports them to clean CSV files (`per_sample_scores.csv`).

### 21. `numpy`
*   **What it is:** The standard mathematical array-processing library.
*   **Where & How it is used:**
    *   Provides mathematical functions and array operations used by metrics (averages, std-dev) and neural network libraries.

### 22. `tqdm`
*   **What it is:** A progress bar library that wraps around loops to display progress.
*   **Where & How it is used:**
    *   Wraps processing loops in evaluation runs to give the user a clear visualization of progress and estimated time of completion.
