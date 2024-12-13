from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api
embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")



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