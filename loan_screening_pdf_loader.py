from langchain_community.document_loaders import PyPDFLoader

def extract_text(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for page in pages:
        
    