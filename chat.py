import sys
import chromadb
from rag_pipeline import ask_meeting_brain, init_db

def get_available_meetings():
    """
    Retrieves the list of unique meeting IDs stored in ChromaDB.
    """
    try:
        collection = init_db()
        results = collection.get(include=["metadatas"])
        meeting_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "meeting_id" in meta:
                meeting_ids.add(meta["meeting_id"])
        return sorted(list(meeting_ids))
    except Exception as e:
        print(f"Error fetching meetings from ChromaDB: {e}")
        return []

def main():
    print("==================================================")
    print("       Meeting Chatbot ('Second Brain') CLI       ")
    print("==================================================")
    
    model = "phi3"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    print(f"Using local model: {model}\n")

    meetings = get_available_meetings()
    selected_meeting = None

    if meetings:
        print("Available meetings in your Second Brain:")
        print("  [0] Global Search (Search across all meetings)")
        for i, m_id in enumerate(meetings, 1):
            print(f"  [{i}] {m_id}")
        print("")
        
        try:
            choice = input("Select a meeting to query (0 to search all, default: 0): ").strip()
            if choice and choice != "0":
                idx = int(choice) - 1
                if 0 <= idx < len(meetings):
                    selected_meeting = meetings[idx]
                    print(f"\n---> Scoped chat activated for meeting: '{selected_meeting}'")
                else:
                    print("\n---> Invalid choice. Defaulting to Global Search.")
            else:
                print("\n---> Global Search activated (searching across all meetings).")
        except ValueError:
            print("\n---> Invalid input. Defaulting to Global Search.")
    else:
        print("No meetings found in the database. Searching globally (empty database).\n")

    print("\nType your questions. Special commands:")
    print("  'switch' : change target meeting")
    print("  'exit'   : close the chatbot\n")
    
    while True:
        try:
            prompt_prefix = f"[{selected_meeting if selected_meeting else 'GLOBAL'}] Ask a question: "
            query = input(prompt_prefix).strip()
            if not query:
                continue
                
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if query.lower() == "switch":
                # Reload meetings list and switch context
                meetings = get_available_meetings()
                if not meetings:
                    print("\nNo meetings in database to switch to.\n")
                    selected_meeting = None
                    continue
                print("\nAvailable meetings:")
                print("  [0] Global Search")
                for i, m_id in enumerate(meetings, 1):
                    print(f"  [{i}] {m_id}")
                
                choice = input("\nSelect meeting (default: 0): ").strip()
                if choice and choice != "0":
                    idx = int(choice) - 1
                    if 0 <= idx < len(meetings):
                        selected_meeting = meetings[idx]
                        print(f"---> Switched to meeting: '{selected_meeting}'\n")
                    else:
                        selected_meeting = None
                        print("---> Switched to Global Search\n")
                else:
                    selected_meeting = None
                    print("---> Switched to Global Search\n")
                continue

            print("\nThinking...")
            answer = ask_meeting_brain(query, meeting_id=selected_meeting, model_name=model)
            print(f"\nAnswer:\n{answer}")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()
