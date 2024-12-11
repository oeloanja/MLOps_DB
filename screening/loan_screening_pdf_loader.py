import fitz
import pandas as pd
import datetime as dt
# import pdfplumber
import openai
import os
from dotenv import load_dotenv

def _get_table(path):
    doc = fitz.open(path)
    for page in doc:
        tables = page.find_tables()
    return tables

def _table2df(path):
    tables = _get_table(path)
    df_list = []
    for table in tables:
        df = table.to_pandas()
        df_list.append(df)
    return df_list

def income(path):
    table = _table2df(path)[0]
    income_ = table.loc[7]
    income = income_[13]
    income_fin = float(income)
    return income_fin

def emp_length(path):
    now = dt.datetime.now()
    now_date = dt.datetime.strftime(now, '%Y-%m-%d')
    now_datetime = dt.datetime.strptime(now_date, '%Y-%m-%d')
    table = _table2df(path)[0]
    # table0 = table[0]
    table_ = table.loc[1]
    start = table_.iloc[1]
    start_datetime = dt.datetime.strptime(start, '%Y년 %m월 %d일')
    during = now_datetime - start_datetime
    during_days = during.days
    emp_length_year = during_days // 365
    return emp_length_year
