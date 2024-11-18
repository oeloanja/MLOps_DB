from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

embedding_model = HuggingFaceEmbeddings(model_name = 'BAAI/bge-m3', model_kwargs = {'device' : 'cpu'}, encode_kwargs={'normalize_embeddings':True},)


def get_vectorstore(docs, collection, dir_path):
    vec_store = Chroma.from_documents(
        collection_name = collection,
        documents = docs,
        embedding = embedding_model,
        persist_directory = dir_path
    )
    return vec_store

def add_new_doc(new_docs, vec_db):
    new_vec_db = vec_db.add_documents(new_docs)
    return new_vec_db

def load_vectorstore(dir_path, collection_name):
    persist_db = Chroma(
        persist_directory = dir_path,
        embedding_function = embedding_model,
        collection_name = collection_name
    )
    return persist_db