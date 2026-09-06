from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
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
    template="Write a joke about {topic}",
    input_variables=["topic"])
prompt2=PromptTemplate(
    template="Explain the following joke {joke} in 2 lines",
    input_variable=["joke"])

joke_gen_chain=RunnableSequence(prompt1 | model | parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2 | model | parser)
})
final_chain=RunnableSequence(joke_gen_chain | parallel_chain)
print(final_chain.invoke({"topic": "Life"}))