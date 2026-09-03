from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("SECRET_KEY"),
)
response = embeddings.embed_query("Hello, how are you?")
print(len(response))
print(response[:5])
