from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("SECRET_KEY"))
response=model.invoke("Capital of India is?")
print(response.content)