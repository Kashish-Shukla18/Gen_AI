from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

model=ChatHuggingFace(llm=endpoint)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a detailed story about {topic}",
    input_variables=["topic"])

prompt2=PromptTemplate(
    template="explain in 2 line this story: {text}",
    input_variables=["text"])


chain1=RunnableSequence(prompt1 | model | parser | prompt2 | model | parser)
print(chain1.invoke({"topic": "India"}))