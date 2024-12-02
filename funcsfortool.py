from langchain_core.tools import tool
import pickle
from langchain.tools import StructuredTool
from pydantic import BaseModel
import VectorStore
# from retriever import Retriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from retriever2 import Retriever
from langchain.tools.retriever import create_retriever_tool
import json


dirpath = './MLOps_chatbot'
collection = 'testdb'
vec_db = VectorStore.load_vectorstore(dir_path = dirpath, collection_name = collection)
ret = Retriever(dirpath, collection, searched = 2)
agent_retirever = ret.as_retriever()


ml = pickle.load(open('tool_ml.pickle', 'rb'))


@tool()
def get_simple_screening(a, b, c, d) -> int:
    """
    주어진 머신러닝 모델을 이용해 간단한 대출심사를 진행.
    대출 심사 요청이 왔을때만 실행.
    예시
    - 나 대출 가능해?
    - 나 연봉 5000인데 대출 가능해?
    - 나 대출 가능한지 봐줘.
    이런 질문들이 들어왔을 때 파라미터를 입력받은 후 이 도구를 사용합니다.
    파라미터가 다 입력된 후 도구 실행 해야합니다.
    누락된 파라미터는 계속 요청하세요.
    Args:
        a:연봉
        b:경력
        c:총부채상환비율, dti
        d:대출 희망 금액
    """
    if b > 10:
        b = 10
    input_data = [[d, c, b, a]]
    prediction = ml.predict(input_data)
    return prediction

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


