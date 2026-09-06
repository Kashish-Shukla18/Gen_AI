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
    template="Write a 2 line paragraph about {topic}",
    input_variables=["topic"])

prompt2=PromptTemplate(
    template="Write a 2 line paragraph about {topic}",
    input_variables=["topic"])

# parallel_chain=RunnableParallel(prompt1 | model | parser, prompt2 | model | parser)  when saame topic is passed to both prmopts
parallel_chain=RunnableParallel({
    "instagram": prompt1 | model | parser,
    "Twitter": prompt2 | model | parser
})
print(parallel_chain.invoke({"topic": "Gen AI"}))