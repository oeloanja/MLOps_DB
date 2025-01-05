'''
document loader
chunking까지 한번에 하도록 구현함
chunking은 데코레이터를 사용해 코드 중복을 방지함
'''


from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
import re
from langchain.schema import Document

def split(func): # split하는 부분. 여기선 charactersplitter 메서드를 사용하지 않고 직접 구현함.
    def wrapper(*args, **kwargs):
        docs = func(*args, **kwargs)
        splitted_doc_list = []
        doc_list = [doc.page_content for doc in docs]
        meta = [doc.metadata for doc in docs]
        for doc in doc_list:
            con_pattern = r'\n\n'
            c_doc = re.sub(con_pattern, '', doc)
            splitted_doc = re.split(r'[.※·－①②③④⑤]', c_doc)
            splitted_doc_list.append(splitted_doc)
        result_doc = [Document(page_content= "\n\n".join(splitted_doc_list), metadata = meta) for splitted_doc_list, meta in zip(splitted_doc_list, meta)] # split 결과를 다시 Document 형태로 바꾸는 작업
        # split 결과는 리스트로 나와 Document 형태로 바꿔줘야함.
        return result_doc
    return wrapper

@split
def docload(file_path):
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    return pages

@split
def dirloader(dir_path):
    loader = DirectoryLoader(dir_path, glob = '*.txt', loader_cls = TextLoader)
    pages = loader.load()
    return pages