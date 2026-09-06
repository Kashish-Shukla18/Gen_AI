from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough
import os

load_dotenv()

def word_count(text:str):
    return len(text.split())

runnable_word_counter=RunnableLambda(word_count)

endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

model=ChatHuggingFace(llm=endpoint)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a report on {topic}",
    input_variables=["topic"])
prompt2=PromptTemplate(
    template="summarise {topic}",
    input_variables=["topic"])

report_gen_chain=RunnableSequence(prompt1 | model | parser)

branch_chain=RunnableBranch(
    (lamda x:len(x.split()))>500 ,RunnableSequence(prompt2 | model | parser),
    RunnablePassthrough())

final_chain=RunnableSequence(report_gen_chain | branch_chain)
result=final_chain.invoke({"topic": "Life"})
print(result)