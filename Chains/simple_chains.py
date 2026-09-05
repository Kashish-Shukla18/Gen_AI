from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

model=ChatHuggingFace(llm=endpoint)

prompt=PromptTemplate(
    template="Write a short story about {topic}",
    input_variables=["topic"])

parser=StrOutputParser()

chain=prompt | model | parser
result=chain.invoke({"topic":"India"})
chain.get_graph().print_ascii()

print(result)
