from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import boto3
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from mangum import Mangum

# Initialize FastAPI



# Bedrock Clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion function (PDF documents)
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector store function using FAISS
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Load the FAISS vector store
def load_vector_store():
    return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

# Prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words with detailed explanations. 
If you don't know the answer, just say that you don't know. Don't make up an answer.
<context>{context}</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# LLaMA3 Model
def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':1024})
    return llm

# Query response function
def get_response(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Input model for the API
class Query(BaseModel):
    question: str
    
app = FastAPI()
handler = Mangum(app)

# Endpoint for updating the vectors (data ingestion and vector store creation)
@app.post("/update_vectors/")
def update_vectors():
    try:
        docs = data_ingestion()
        get_vector_store(docs)
        return {"message": "Vector store updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for querying the vector store
@app.post("/query/")
def query_vector_store(query: Query):
    try:
        # Load the FAISS index
        vectorstore_faiss = load_vector_store()

        # Load the LLaMA model
        llm = get_llama3_llm()

        # Get the response
        response = get_response(llm, vectorstore_faiss, query.question)

        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

## setting up the lambda to be called


# Run the app using uvicorn (from terminal)
# uvicorn main:app --reload
