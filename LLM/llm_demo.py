from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()

llm=ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("SECRET_KEY"))
response=llm.invoke("Who is India's Prime Minister and what is hig age??")
print(response.content)