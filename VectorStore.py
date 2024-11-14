from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def embedding(docs):
    pass

def get_vectorstore(docs, embedding_func, dir_path):
    vec_store = Chroma(
        
    )