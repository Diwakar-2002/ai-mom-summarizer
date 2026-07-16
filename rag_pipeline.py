import os
import json
import requests
import chromadb

def init_db(db_path="./meeting_db"):
    """
    Initializes a persistent ChromaDB client saving to the specified local folder.
    Creates or gets a collection called 'meeting_memory'.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="meeting_memory")
    return collection

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Chunks the input text into segments of roughly `chunk_size` words
    with `overlap` words of overlap.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Ensure we progress even if chunk_size <= overlap
        step = max(1, chunk_size - overlap)
        start += step
    return chunks

def get_ollama_embedding(text, model="nomic-embed-text", host="http://localhost:11434"):
    """
    Calls the local Ollama API to generate embeddings using 'nomic-embed-text'.
    """
    url = f"{host}/api/embeddings"
    payload = {"model": model, "prompt": text}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        return res_json.get("embedding")
    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Could not connect to Ollama server.")
        print("Please make sure Ollama is running on your machine.")
        print("You can run Ollama or check its status. Also ensure the model is pulled:")
        print(f"  ollama pull {model}\n")
        raise
    except Exception as e:
        print(f"Error fetching embedding from Ollama: {e}")
        raise

def store_meeting(meeting_id, transcript, summary, db_path="./meeting_db"):
    """
    Chunks the transcript, generates embeddings for chunks/summary, 
    and stores them in a local ChromaDB collection.
    """
    collection = init_db(db_path)
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    # 1. Embed and prepare the summary
    if summary:
        summary_id = f"{meeting_id}_summary"
        print(f"Generating embedding for meeting summary: {meeting_id}...")
        try:
            summary_emb = get_ollama_embedding(summary)
            ids.append(summary_id)
            embeddings.append(summary_emb)
            documents.append(summary)
            metadatas.append({"meeting_id": meeting_id, "type": "summary"})
        except Exception as e:
            print(f"Skipping summary due to embedding failure: {e}")
            
    # 2. Chunk and embed the transcript
    chunks = chunk_text(transcript, chunk_size=400, overlap=50)
    print(f"Split transcript into {len(chunks)} chunk(s). Generating embeddings...")
    for i, chunk in enumerate(chunks):
        chunk_id = f"{meeting_id}_chunk_{i}"
        try:
            chunk_emb = get_ollama_embedding(chunk)
            ids.append(chunk_id)
            embeddings.append(chunk_emb)
            documents.append(chunk)
            metadatas.append({
                "meeting_id": meeting_id,
                "type": "transcript_chunk",
                "chunk_index": i
            })
        except Exception as e:
            print(f"Skipping chunk {i} due to embedding failure: {e}")
            
    if ids:
        # Upsert embeddings to ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully stored meeting '{meeting_id}' in ChromaDB collection 'meeting_memory'.")
    else:
        print("No content was successfully embedded. ChromaDB was not updated.")

def ask_meeting_brain(user_query, meeting_id=None, model_name="phi3", db_path="./meeting_db", host="http://localhost:11434"):
    """
    Queries ChromaDB for the top 3 most relevant segments to a user query
    and constructs a RAG prompt for local Ollama completion.
    Supports filtering by a specific meeting_id.
    """
    collection = init_db(db_path)
    
    # Convert query to embedding
    print(f"Generating embedding for query: '{user_query}'...")
    query_emb = get_ollama_embedding(user_query)
    
    # Query ChromaDB
    print("Querying ChromaDB for top 3 matching documents...")
    query_args = {
        "query_embeddings": [query_emb],
        "n_results": 3
    }
    if meeting_id:
        query_args["where"] = {"meeting_id": meeting_id}
        print(f"Filtering search results to meeting: '{meeting_id}'")
        
    results = collection.query(**query_args)

    
    retrieved_chunks = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    
    if not retrieved_chunks:
        return "No relevant context found in the database. Please ingest some meetings first."
        
    print(f"Retrieved {len(retrieved_chunks)} relevant chunk(s):")
    for i, meta in enumerate(retrieved_metadata):
        print(f"  [{i+1}] Source Meeting: {meta.get('meeting_id')} (Type: {meta.get('type')})")
        
    # Construct RAG prompt
    context = "\n---\n".join(retrieved_chunks)
    rag_prompt = f"You are a corporate assistant. Based ONLY on the following context, answer the user's question.\n\nContext:\n{context}\n\nQuestion:\n{user_query}"
    
    # Post query prompt to Ollama generate
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": rag_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    print(f"Sending prompt to local model '{model_name}'...")
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        return res_json.get("response", "")
    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Could not connect to Ollama server.")
        print("Please make sure Ollama is running.")
        print(f"You may need to pull the model: ollama pull {model_name}\n")
        raise
    except Exception as e:
        print(f"Error during model generation: {e}")
        raise

if __name__ == "__main__":
    # Mock meeting data to test pipeline
    mock_meeting_id = "springfield_council_meeting_2026"
    mock_summary = (
        "In this Springfield Council meeting, the council approved a $500,000 budget for the new "
        "recreation center. They also discussed bylaws regarding parking regulations and assigned the "
        "task of updating the town website to Ethan Wiebe by next Friday."
    )
    mock_transcript = (
        "Mayor Quimby: Welcome everyone to the Springfield council meeting. Today we have a few major items. "
        "First, let's discuss the funding for the Springfield Recreation Center. We are proposing a budget "
        "of five hundred thousand dollars. Is there a motion to carry? "
        "Councilor Hibbert: Yes, I motion to approve the budget. "
        "Councilor Wiggum: I second that. "
        "Mayor Quimby: The motion is carried. $500,000 is officially approved for the recreation center. "
        "Next, we need to address parking regulations on Main Street. The current bylaw is outdated. "
        "We need to review it by next month. "
        "Ethan Wiebe: I can start drafting a revised parking proposal. "
        "Mayor Quimby: Great, Ethan. Also, the town website needs to be updated with these minutes. "
        "Can you handle uploading it? "
        "Ethan Wiebe: Yes, I will update the town website with the new bylaws by next Friday. "
        "Mayor Quimby: Perfect. That wraps up today's business. Meeting adjourned."
    )
    
    print("--- STEP 1: Ingestion & Embedding ---")
    try:
        store_meeting(mock_meeting_id, mock_transcript, mock_summary)
        print("Ingestion complete.\n")
    except Exception as e:
        print(f"Ingestion failed: {e}\n")
        
    print("--- STEP 2: Retrieval & Query answering ---")
    query = "How much money was approved for the recreation center, and who is updating the town website?"
    try:
        # Default to phi3 model. The user can also try other models they have.
        answer = ask_meeting_brain(query, model_name="phi3")
        print("\n=== QUESTION ===")
        print(query)
        print("\n=== ANSWER ===")
        print(answer)
        print("================\n")
    except Exception as e:
        print(f"Query execution failed: {e}")
