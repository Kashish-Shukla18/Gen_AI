from langchain_community.document_loadersf import PyPDFLoader

loader=PyPDFLoader("dl-curriculum.pdf")
documents=loader.load()
print(documents)
