'''
벡터DB를 기반으로 retriever를 만드는 역할을 하는 모듈
'''

from langchain_chroma.vectorstores import Chroma
import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_transformers import LongContextReorder

reorder = LongContextReorder()

class Retriever(Chroma):
    def __init__(self, dir_path, collection_name, searched:int):
        super().__init__()
        self.vec_db = VectorStore.load_vectorstore(dir_path, collection_name)
        self.searched = searched
        
    def as_retriever(self):
        vec_db = self.vec_db
        retriever = vec_db.as_retriever(search_kwargs = {'k' : self.searched})
        return retriever
    
    def get_relevant_documents(self, input):
        retriever = self.as_retriever()
        result = retriever.invoke(input)
        reordered = reorder.transform_documents(result)
        return reordered