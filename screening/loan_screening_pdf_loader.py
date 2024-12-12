import fitz
import pandas as pd
import datetime as dt
# import pdfplumber
import boto3
import os
from urllib.parse import urlparse
import io
from dotenv import load_dotenv

load_dotenv()
ID = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['OPENAI_API_KEY'] = ID
PW = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = PW


def parse_s3_url(url):
    parsed = urlparse(url)
    bucket = parsed.netloc.split('.')[0]  # billit-bucket
    key = parsed.path.lstrip('/')         # uploads/1733986366870-소득.pdf
    return bucket, key

def _get_table(path):
    bucket_name, key = parse_s3_url(path)
    s3_client = boto3.client('s3',
                             aws_access_key_id = ID,
                             aws_secret_access_key = PW)
    buffer = io.BytesIO()
    s3_client.download_fileobj(bucket_name, key, buffer)
    buffer.seek(0)
    doc = fitz.open(stream=buffer, filetype="pdf")
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
