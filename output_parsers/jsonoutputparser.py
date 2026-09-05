from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation",)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
template=PromptTemplate(
    template="Write a detailed report on any fictional country in the world \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions":parser.get_format_instructions() })

# prompt=template.format()
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)

chain=template | model | parser
result=chain.invoke({})
print(result)
