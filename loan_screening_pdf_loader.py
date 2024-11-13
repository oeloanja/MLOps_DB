import pdfplumber
import pandas as pd

def extract_text(file_path):
    df = pd.DataFrame()
    with pdfplumber.open(file_path) as pdf:
        for page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_table()
            if tables:
                
