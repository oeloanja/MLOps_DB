from langchain_core.tools import BaseTool
import pickle
from pydantic import BaseModel, Field
import VectorStore
from retriever import Retriever
from langchain.tools.retriever import create_retriever_tool
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
import pandas as pd




dirpath = './MLOps_chatbot'
collection = 'testdb'
vec_db = VectorStore.load_vectorstore(dir_path = dirpath, collection_name = collection)
ret = Retriever(dirpath, collection, searched = 2)
agent_retirever = ret.as_retriever()



def retrieve():
    retriever_tool = create_retriever_tool(
        agent_retirever,
        name = 'local_retriever',
        description = '''금융 용어, 대출 방법, 투자 방법에 관한 문서들을 검색할 수 있는 검색기 도구입니다. 
        금융 용어, 대출 방법, 투자 방법에 대한 질문이 들어오면 사용합니다.
        예시
        - 자료열람요구권이 뭐야?
        - 나 대출 받는 방법이 궁금해.
        - 투자를 하고싶은데 어떻게 해
        이런 질문에는 이 local retriever를 써 관련된 문서를 찾은 후 그 문서를 기반으로 답하세요.'''
    )
    return retriever_tool
