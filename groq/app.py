import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain   #when we create a document loader we need retrieval chain also 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain #Doc retreival chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv

load_dotenv()

##load the Groq aPI key

groq_api_key=os.environ["GROQ_API_KEY"]


# creating a session state to understand the code flow

if "vector" not in st.session_state:
    print("in vector")

    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader= WebBaseLoader("https://docs.smith.langchain.com/")
    print("After webbaseloader")
    #loads whole content from website
    st.session_state.docs=st.session_state.loader.load()
    print("After load")
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    print("After text splitter")
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:100])
    print("After final doxs")
    #st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    print("After vectors")


st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Gemma-7b-it")
print("After model call")



prompt=ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most acurate response based on the questions
<context>
{context}
Questions:{input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
prompt=st.text_input("Input your prompt here")

if prompt:
    response=retrieval_chain.invoke({"input":prompt})
   # print("Response time :", time.process_time()-start)
    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("document similarity search"):
        #find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------------------------------------")
        
        
        

