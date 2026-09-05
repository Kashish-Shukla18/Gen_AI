from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("SECRET_KEY"))
chat_history=[
    SystemMessage(content="You are a helpful assistant."),]

chat_history=[]
while True:
    input_text=input("You: ")
    chat_history.append(HumanMessage(content=input_text))
    if input_text == 'exit':
        break
    response=model.invoke(input_text)
    chat_history.append(AIMessage(content=response.content))
    print("Assistant: ", response.content)

print(chat_history)