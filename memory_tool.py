import chromadb
import uuid

class Memory:
    def __init__(self, db_path="jarvis_memory"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="long_term_memory")
        print("Memory Core Initialised.")

    def add_memory(self, text: str):
        memory_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            ids=[memory_id]
        )
        print(f"Memory added with ID: {memory_id}")

    def retrieve_memory(self, query: str, n_results: int = 1) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results 
        )
        if results and results['documents'][0]:
            return results['documents'][0][0]
        else:
            return "No relevant memory found."

if __name__ == '__main__':
    memory = Memory()
    
    #Add some initial memories to the database
    print("\n--- Adding initial memories ---")
    memory.add_memory("The user's name is Joe and their goal is to build a jarvis-style AI.")
    memory.add_memory("We are building Jarvis on an Arch Linux machine with an AMD ROCm-compatible GPU.")

    # Testing retrieving a memory
    print("\n--- Testing memory retrieval ---")
    query = "What is the user's goal?"
    retrieved = memory.retrieve_memory(query)

    print(f"\nQuery: '{query}'")
    print(f"Retrieved: '{retrieved}'")
