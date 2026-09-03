from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Hello, how are you?",
    "The sky is blue.",
    "The sky is red.",
    "The sky is green.",
    "The sky is yellow.",
    "The sky is orange.",
    "The sky is purple.",
    "The sky is pink.",
    "The sky is brown.",
    "The sky is gray.",
]

document_embeddings = embeddings.embed_documents(documents)
query = "The sky is blue."
query_embedding = embeddings.embed_query(query)

scores = [
    np.dot(query_embedding, doc_emb)
    / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
    for doc_emb in document_embeddings
]

best_idx = int(np.argmax(scores))
print(f"Query: {query}")
print(f"Best match: {documents[best_idx]}")
print(f"Similarity: {scores[best_idx]:.4f}")
print("\nAll scores:")
for doc, score in zip(documents, scores):
    print(f"  {score:.4f}  {doc}")
