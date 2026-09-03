from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    pipeline_kwargs={"temperature": 0.5, "max_length": 100},
)

model=ChatHuggingFace(llm=llm)
response=model.invoke("Capital of India is?")
print(response.content)