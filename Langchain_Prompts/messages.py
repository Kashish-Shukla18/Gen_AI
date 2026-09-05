from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("SECRET_KEY"))
messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
response=model.invoke(messages)
messages.append(AIMessage(content=response.content))
print(messages)