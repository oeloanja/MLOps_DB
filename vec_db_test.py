import VectorStore
from Retriever import retriever
import DocLoader

docs = DocLoader.docload('C:/Users/user/Downloads/가계대출상품설명서.pdf')
vec_db = VectorStore.get_vectorstore(docs, collection = 'testdb', embedding_func=VectorStore.embedding(docs), dir_path = './MLOps_chatbot')
