from langchain_core.tools import tool
import pickle
from langchain.tools import StructuredTool

import VectorStore
from Retriever import retriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain.tools.retriever import create_retriever_tool


dirpath = './MLOps_chatbot'
collection = 'testdb'
vec_db = VectorStore.load_vectorstore(dir_path = dirpath, collection_name = collection)
agent_retirever = retriever(vec_db, searched = 3)


ml = pickle.load(open('tool_ml.pickle', 'rb'))

@tool
def get_simple_screening() -> str:
    """
    주어진 머신러닝 모델을 이용해 간단한 대출심사를 진행.
    대출 심사 요청이 왔을때만 실행.
    income, job_duration, dti, loan_amnt가 다 입력된 후 도구 실행 해야합니다
    Args:
        income:연봉
        job_duration:경력
        dti:총부채상환비율
        loan_amnt:대출 받고자 하는 돈
    """
    income:float
    job_duration:int
    dti:float
    loan_amnt:float
    if job_duration > 10:
        job_duration = 10
    input_data = [[loan_amnt, dti, job_duration, income]]
    prediction = ml.predict(input_data)
    return str(prediction)

def retrieve():
    retriever_tool = create_retriever_tool(
        agent_retirever,
        name = 'local retriever',
        description = 'you must use this tool to search information'
    )
    return retriever_tool


