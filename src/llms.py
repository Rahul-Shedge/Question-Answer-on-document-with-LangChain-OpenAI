import tiktoken
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


   
# 
def ask_and_get(db,question,api_key,k=3):
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': k})
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1,openai_api_key = api_key)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(question)
    return answer

def calculate_embedding_cost(text):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(encoding.encode(page.page_content)) for page in text])
    return total_tokens,total_tokens/1000*0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

