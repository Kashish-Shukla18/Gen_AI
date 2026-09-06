from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

model=ChatHuggingFace(llm=endpoint)
parser=StrOutputParser()

prompt=PromptTemplate(
    template="summarise the following text in 3 lines: {text}",
    input_variables=["text"]
)

loader=TextLoader('cricket.txt',encoding='utf-8')
documents=loader.load()

chain=prompt | model | parser
result=chain.invoke({"text":documents[0].page_content})
print(result)


# print(documents[0].page_content)
# print(documents[0].metadata)