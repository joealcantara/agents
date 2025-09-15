import chromadb
import json

client = chromadb.PersistentClient(path='jarvis_memory')

collection = client.get_or_create_collection(name="long_term_memory")

print("Retrieving all memories from the database...")
all_memories = collection.get()

print(json.dumps(all_memories, indent=2))

print("\n--- Starting Deduplication Process ---")
unique_documents = set()
ids_to_delete = []

for memory_id, document in zip(all_memories['ids'], all_memories['documents']):
    if document in unique_documents:
        ids_to_delete.append(memory_id)
    else:
        unique_documents.add(document)

print(f"Found {len(ids_to_delete)} duplicate memories to delete.")

if ids_to_delete:
    collection.delete(ids=ids_to_delete)
    print("Successfully deleted duplicate memories.")
else:
    print("No duplicate memories found.")
