from typing import List
import langchain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.document_transformers import LongContextReorder

class retriever():
    def __init__(self, vec_db, searched):
        self.vec_db = vec_db
        self.searched = searched
        
    def __call__(self, input):
        query = self.preprocess_query(input)
        return self.get_docs(query)

    def preprocess_query(self, input):
        return input.get("input", "")

    def get_retriever(self):
        vec_store = self.vec_db
        retriever = vec_store.as_retriever(search_kwargs = {"k" : self.searched})
        return retriever
        
    def reorder(func):
        def wrapper(*args, **kwargs):
            commander = LongContextReorder()
            searched_docs = func(*args, **kwargs)
            reordered_docs = commander.transform_documents(searched_docs)
            return reordered_docs
        return wrapper
        
    @reorder
    def get_docs(self, query):
        retriever = self.get_retriever()
        if not isinstance(query, str):  # query가 문자열인지 확인
            raise ValueError(f"Expected a string query, but got {type(query).__name__}")
        docs = retriever.get_relevant_documents(query)
        return docs