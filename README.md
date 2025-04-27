# RAG Document Q&A System

This repository contains a Streamlit application that implements a Retrieval-Augmented Generation (RAG) system for question answering on PDF documents. The system uses Groq's LLM API and HuggingFace embeddings to provide accurate responses based on document content.

## Features

- PDF document processing and embedding
- Vector-based similarity search using FAISS
- Integration with Groq's Llama3-8b-8192 model
- Interactive Q&A interface built with Streamlit
- Document context visualization for transparency

## How It Works

The application follows these steps:
1. Loads a PDF document ("Attention.pdf")
2. Splits the document into manageable chunks
3. Creates vector embeddings using HuggingFace's all-MiniLM-L6-v2 model
4. Stores embeddings in a FAISS vector database
5. Retrieves relevant document sections based on user queries
6. Processes the query and context through Groq's LLM
7. Returns an answer with supporting document sections

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- Groq API access

## Installation

```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt
```

## Usage

1. Add your API keys to the environment variables or create a `.env` file
2. Place your PDF document in the project directory (default: "Attention.pdf")
3. Run the Streamlit app:

```bash
streamlit run main.py
```

4. Enter your query in the text input field
5. Click "Document Embedding" to process the document
6. View the answer and explore similar document sections in the expander

## API Keys

The application requires the following API keys:
- GROQ_API_KEY: For accessing Groq's LLM
- HF_TOKEN: For HuggingFace embeddings
- LANGCHAIN_API_KEY: For LangChain integration
- GOOGLE_API_KEY: For additional services

## Customization

You can modify the following parameters:
- Change the PDF document by updating the filename in `PyPDFLoader()`
- Adjust chunk size and overlap in `RecursiveCharacterTextSplitter()`
- Select a different LLM model by changing the `model_name` parameter
- Customize the prompt template for different response styles

## Acknowledgments

This project uses several open-source libraries and APIs:
- LangChain for orchestrating the RAG pipeline
- Groq for LLM inference
- HuggingFace for embeddings
- FAISS for vector similarity search
- Streamlit for the web interface
