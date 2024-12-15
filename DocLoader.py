from langchain_community.document_loaders import PDFMinerLoader, PyMuPDFLoader, TextLoader
import re
from pdfminer.layout import LTTextBox, LTImage
from langchain.schema import Document

def split(func):
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
        result_doc = [Document(page_content= "\n\n".join(splitted_doc_list), metadata = meta) for splitted_doc_list, meta in zip(splitted_doc_list, meta)]
        return result_doc
    return wrapper

@split
def docload(file_path):
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    return pages

@split
def textloader(file_path):
    loader = TextLoader(file_path)
    pages = loader.load()
    return pages