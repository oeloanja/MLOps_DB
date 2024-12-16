import VectorStore
import DocLoader

docs = DocLoader.docload('C:/Users/user/Downloads/Billit_사용자_설명서.pdf')
vec_db = VectorStore.get_vectorstore(docs, collection = 'testdb', dir_path = './MLOps_chatbot')

docs2 = DocLoader.textloader('./financial_vocab.txt')
vec_db2 = VectorStore.get_vectorstore(docs2, collection = 'findb', dir_path = './fin_vocab')
