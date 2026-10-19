import chromadb

client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_collection("faq")

data = collection.get(include=["documents", "metadatas", "embeddings"])
print("Total stored:", len(data["ids"]))

for i in range(min(2, len(data["ids"]))):
    print("\n--- Entry", i, "---")
    print("id        :", data["ids"][i])
    print("text      :", data["documents"][i])
    print("metadata  :", data["metadatas"][i])
    print("vector    :", len(data["embeddings"][i]), "numbers, starting", data["embeddings"][i][:5])