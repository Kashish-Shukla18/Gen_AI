from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
import os

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
query="The sky is blue."

query_embedding = embeddings.embed_query(query)

cosine_similarity = np.dot(query_embedding, document_embeddings[0]) / (np.linalg.norm(query_embedding) * np.linalg.norm(document_embeddings[0]))
print(cosine_similarity)