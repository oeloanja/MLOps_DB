from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/testdb"

base = declarative_base()
engine = create_engine(DATABASE_URL)
sessionlocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)

def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()