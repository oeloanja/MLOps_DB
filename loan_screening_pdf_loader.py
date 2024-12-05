import fitz
import pandas as pd
# import pdfplumber

def _get_table(path):
    doc = fitz.open(path)
    for page in doc:
        tables = page.find_tables()
    return tables

def table2df(path):
    tables = _get_table(path)
    df_list = []
    for table in tables:
        df = table.to_pandas()
        df_list.append(df)
    return df_list

def income(path):
    table = table2df(path)[0]
    income_ = table.loc[7]
    income = income_[13]
    income_fin = float(income)
    return income_fin

# import tabula
# import pandas as pd

# def extract_tables_from_pdf(file_path):
#     # Accepts file path as an argument
#     tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
#     # Extract tables from PDF document using tabula library
#     dataframes = []
#     for table in tables:
#         df = pd.DataFrame(table)
#         dataframes.append(df)
#     return dataframes
