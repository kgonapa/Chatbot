import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain   #when we create a document loader we need retrieval chain also 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain #Doc retreival chain
from langchain_objectbox.vectorstores import objectbox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

##load the Groq API key

os.environ['OPEN_API_KEY']=os.getenv("OPENAPI_API_KEY")

groq_api_key=os.environ("GROQ_API_KEY")

st.title("Objectbox VectorstoreDB with Llama3 demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

##Vecto Embedding and objectbox vectorstore db

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_document(st.session_state.docs[:20])
        st.session_state.vectors=objectbox.from_documents(st.session_state.final_documents,st.session_state.embeddings)
