from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableParallel

load_dotenv()

endpoint1=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

endpoint2=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

model1=ChatHuggingFace(llm=endpoint1)

model2=ChatHuggingFace(llm=endpoint2)

prompt1=PromptTemplate(
    template="Generate short and simple notes about {topic}",
    input_variables=["topic"])

prompt2=PromptTemplate(
    template="Generate a 5 questions and answers quiz about {text}",
    input_variables=["text"])

prompt3=PromptTemplate(
    template="Mege the following into a single document: {notes} {quiz}",
    input_variables=["notes", "quiz"])

parser=StrOutputParser()

# Parallel Chain
parallel_chain=RunnableParallel({
    "notes":prompt1 | model1 | parser,
    "quiz":prompt2 | model2 | parser
})

# Merging chain
merge_chain = prompt3 | model1 | parser

chain parallel_chain | merge_chain
result=chain.invoke({"topic":"India"})
print(result)
chain.get_graph().print_ascii()