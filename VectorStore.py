from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def embedding(docs):
    pass

def get_vectorstore(docs, collection, embedding_func, dir_path):
    vec_store = Chroma.from_documents(
        collection_name = collection,
        documents = docs,
        embedding = embedding_func,
        persist_directory = dir_path
    )
    return vec_store

def add_new_doc(new_docs, vec_db):
    new_vec_db = vec_db.add_documents(new_docs)
    return new_vec_db
