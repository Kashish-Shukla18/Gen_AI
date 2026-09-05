from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation",)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str=Field(description="The name of the person")
    age:int=Field(description="The age of the person")
    email:EmailStr=Field(description="The email of the person")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="Generate the name,age and email of a person from {place}",
    input_variables=["place"],
    partial_variables={"format_instructions":parser.get_format_instructions() })

# prompt=template.invoke({"place":"India"})
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result) 

chain=template | model | parser
result=chain.invoke({"place":"India"})
print(result)