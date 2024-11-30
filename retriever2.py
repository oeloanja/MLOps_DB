from langchain_chroma.vectorstores import Chroma
import VectorStore

class Retriever(Chroma):
    def __init__(self, dir_path, collection_name, searched:int):
        super().__init__()
        self.vec_db = VectorStore.load_vectorstore(dir_path, collection_name)
        self.searched = searched
        
    def as_retriever(self):
        vec_db = self.vec_db
        retriever = vec_db.as_retriever(search_kwargs = {'k' : self.searched})
        return retriever