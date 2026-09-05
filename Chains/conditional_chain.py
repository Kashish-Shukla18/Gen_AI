# feedback analyse(+ve/-ve) for positive give positive reply for negative accordingly

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
import os
from langchain_core.runnables import RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )

model=ChatHuggingFace(llm=endpoint)
parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="The sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)
prompt1=PromptTemplate(
    template="Classify the sentiment of following text to positive or negative: {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions":parser2.get_format_instructions()})


classifier_chain=prompt1 | model | parser2

prompt2=PromptTemplate(
    template="Write a positive reply for the following feedback: {feedback}",
    input_variables=["feedback"])

prompt3=PromptTemplate(
    template="Write a negative reply for the following feedback: {feedback}",
    input_variables=["feedback"])


brach_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        prompt2 | model | parser
    ),
    (
        lambda x: x.sentiment == "negative",
        prompt3 | model | parser
    ),
    prompt3 | model | parser
)

chain=classifier_chain | brach_chain
result=chain.invoke({"feedback":"The product is great!"})
print(result)