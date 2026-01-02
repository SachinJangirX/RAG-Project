# medical RAG application

## overview 
Retrieval-Augmented Generation system using Langchain, Pinecone, and OpenAI.

## architecture 
PDF -> Chunking -> Embeddings -> Pinecone -> Retriever -> LLM -> Flask UI


## setup 
pip install -r requirements.txt
cp .env

## run 
python store_index.py
python app.py

## tech stack 
-Langchain
-Pinecone
-OpenAI
-Flask
-HuggingFace Embeddings


# AWS  CICD Deployment with Github Actions

# login to AWS console 

# create IAM user for deployment

