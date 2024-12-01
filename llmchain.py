from langchain.chains import RetrievalQA
from retriever2 import Retriever
from langchain_ollama import ChatOllama


dir_path = './MLOps_chatbot'
collection = 'testdb'


class GetChain():
    def __init__(self, dir_path, collection, searched, model):
        self.ret_obj = Retriever(dir_path, collection, searched)
        self.model = model
    
    def get_chain(self):
        chain = RetrievalQA.from_chain_type(llm = self.model,
                            retriever = self.ret_obj.as_retriever(),
                            return_source_documents = True)
        return chain