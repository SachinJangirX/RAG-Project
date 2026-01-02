from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract the data from the PDF files
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """given a list of document objects, return a new list of document objects with only the 'source' in metadata and the original page_content"""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source": src}
            )
        )

    return minimal_docs

# split the data into the chunks 
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# download the embedding from hugging face

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings