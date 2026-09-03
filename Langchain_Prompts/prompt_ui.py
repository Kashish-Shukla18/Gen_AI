from pathlib import Path
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import load_prompt

load_dotenv()

st.header("Research Tool")
model=ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("SECRET_KEY"))
# prompt = st.text_input("Enter your prompt")

# if st.button("Generate"):
#     llm=ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("SECRET_KEY"))
#     response=llm.invoke(prompt)
#     st.write(response.content)


paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template = load_prompt('template.json')



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)