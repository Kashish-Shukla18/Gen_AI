# Prompt->llm->parser->passtrough / runnable lambda function(to calculate length of joke) ->print joke+ length of joke
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

prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"])

joke_gen_chain=RunnableSequence(prompt | model | parser)
parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_count)})
final_chain=RunnableSequence(joke_gen_chain | parallel_chain)
result=final_chain.invoke({"topic": "Life"})
final_result="""{} \n word count -{}""".format(result["joke"],result["word_count"])
print(final_result)