# Importing libraries
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain                # for RAG
from langchain.chains import create_retrieval_chain                                        # for RAG
from langchain_community.document_loaders import PyPDFLoader

# Credential and Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ChatBot with LLM"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3jjlk9rl6FBUASv21T1aBAFo_h_R6rTk"
os.environ["HF_TOKEN"] = "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"
os.environ["GROQ_API_KEY"] = "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"

# Assigning the LLM
llm = ChatGroq(model_name = "Llama3-8b-8192")

# Assigning the template
prompt = ChatPromptTemplate.from_template(                                   # {context} --> will be dynamically filled later with actual information
'''           
    Answer the question based on the provided context only.
    Kindly provide the most accurate response based on the question.
    <context>
    {context}                                                   
    <context>
    Question: {input}
'''
)

# Main function
def create_vector_embedding():
    if "vectors" not in st.session_state:                                                                             # st.session_state is used to make sure that app should have memory wrt chat
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")                          # Saves this model inside st.session_state.embeddings
        st.session_state.loader = PyPDFLoader("Attention.pdf")                                                         
        st.session_state.docs = st.session_state.loader.load()                                                        # st.session_state.docs now holds the raw contents (text) from the PDF.
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)       
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) # Only the first 50 documents are used
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)# Saves the FAISS vector store into st.session_state.vector    
        
st.title("RAG Document Q&A With Groq And HuggingFace")

user_prompt = st.text_input("Enter your query from the reserch paper.")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector database is ready")
    
import time
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()                           # .as_retriever(): Converts the FAISS vector store into a retriever, which allows you to search for the most relevant documents based on a query.
    retriever_chain = create_retrieval_chain(retriever, document_chain)           # This chain will allow for both retrieving documents and processing them with an LLM.
    
    start = time.process_time()
    response = retriever_chain.invoke({"input" : user_prompt})
    print(f"Response time: {time.process_time() - start}")
    
    st.write(response["answer"])
    
    # With Streamlit expander
    with st.expander("Documnet Similarity Search"):
        for i, j in enumerate(response['context']):
            st.write(j.page_content)
            st.write("_____________________________________________________________")