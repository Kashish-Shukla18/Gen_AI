from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage

chat_template=ChatPromptTemplate([
    ('system','You are a helpful {domain} assistant.'),
    ('human','Explain me in simple terms, what is {topic}?')])
prompt=chat_template.invoke({"domain":"AI","topic":"AI"})
print(prompt)
print(prompt.content)