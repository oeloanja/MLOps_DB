from sqlalchemy import Column, String
from testapidb import base

class borrow_user(base):
    __tablename__ = 'borrow_user'
    user_id = Column(String(255), primary_key = True, index = True)
    username = Column(String(20), nullable = False)
    email = Column(String(255), nullable = False)