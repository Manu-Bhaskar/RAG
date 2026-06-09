import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

DATA_PATH = "./data/docs"
DB_PATH = "./vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def create_vector_db():
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    print("Loading documents...")

    loaders = [
        DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    if not documents:
        print("No documents found!")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )

    print("Vector DB created!")

if __name__ == "__main__":
    create_vector_db()
