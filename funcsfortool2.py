'''
agent가 쓸 툴들을 구현해놓은 모듈
SimpleScreening은 간이대출심사 툴
retrieve는 서비스 사용법 알려주는 툴을 위한 모듈
fin_retirever는 금융 어휘 사전 툴을 위한 모듈

SimpleScreening 모듈은 getdti 모듈을 통해 dti를 구함
그 후 loan_amnt, dti, issue_d_peroid, annual_income을 간이대출심사 모델에 입력
그 결과값을 통해 대출 가능 여부 확인

retrieve는 서비스 사용방법 질의응답 툴을 위한 모듈
agentic rag 개념을 활용해 구현

fin_retirever는 간단한 금융 어휘를 알려주는 툴을 위한 모듈
이 역시 agentic rag 개념을 활용해 구현현
'''

from langchain_core.tools import BaseTool
import pickle
from pydantic import BaseModel, Field
import VectorStore
from retriever import Retriever
from langchain.tools.retriever import create_retriever_tool
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
import pandas as pd
import getdti
from flask import request

import requests


dirpath = './MLOps_chatbot'
collection = 'testdb'
vec_db = VectorStore.load_vectorstore(dir_path = dirpath, collection_name = collection)
ret = Retriever(dirpath, collection, searched = 2)
agent_retirever = ret.as_retriever()

dir2 = './fin_vocab'
collection2 = 'findb'
vec_db2 = VectorStore.load_vectorstore(dir_path = dir2, collection_name = collection2)
ret2 = Retriever(dir2, collection2, searched = 2)
agent_retirever2 = ret2.as_retriever()


ml = pickle.load(open('ml_for_tool.pkl', 'rb')) #간이대출심사 모델

class ScreeningInput(BaseModel):
   annual_income:int = Field(..., description = "사용자가 입력한 질문에 있는 연봉. 혹은 사용자가 입력한 질문에 있는 소득.")
   period:int = Field(..., description = "사용자가 입력한 질문에 있는 마지막 대출 후 경과한 햇수. 단위는 무조건 년으로 받음.")
   loan_amount:int = Field(..., description = "사용자가 입력한 질문에 있는 대출 희망 금액.")

class SimpleScreening(BaseTool):
    name = 'simple_screening'
    description = """
    주어진 머신러닝 모델을 이용해 간단한 대출심사를 진행.
    대출 심사 요청이 왔을때만 실행.
    예시
    - 나 대출 가능해?
    - 나 대출 가능한지 봐줘.
    이런 질문들이 들어왔을 때 파라미터를 입력받은 후 이 도구를 사용합니다.
    누락된 파라미터가 있을 땐 체인 종료 후 사용자에게 누락된 파라미터를 요청하세요.
    TypeError시 사용자에게 다시 입력 받으세요.
    """
    args_schema : Type[BaseModel] = ScreeningInput


    def _run(self, annual_income:int, period:int, loan_amount:int, run_manager: Optional[CallbackManagerForToolRun] = None):
        pn_data = request.get_json() #getdti 모듈을 통해 dti를 구하려면 사용자의 전화번호가 필요함. 이 경우 세션이 아니라 로컬 스토리지를 통해 로그인 정보를 확인해서 서버에 전화번호를 요청함.
        user_pn = pn_data['user_pn'] 
        dti = getdti.calculate_dti(user_pn, annual_income) #사용자 전화번호는 DB에서 dti를 구할 때 필요한 데이터들을 찾을 때 씀.
        input_data = pd.DataFrame([{'loan_amnt' : loan_amount, 'dti' : dti, 'issue_d_period' : period, 'annual_inc' : annual_income}])
        prediction = ml.predict(input_data)
        screening_result = ''
        if prediction[0] == 0:
            screening_result = '대출이 가능합니다.'
        elif prediction[0] == 1:
            screening_result = '대출이 가능하지만 금리가 조금 더 높을 수 있어요.'
        elif prediction[0] == 2:
            screening_result = '아쉽지만 탈락입니다.'
        return screening_result

def retrieve():
    retriever_tool = create_retriever_tool(
        agent_retirever,
        name = 'local_retriever',
        description = '''서비스 사용법, 대출 방법, 투자 방법에 관한 문서들을 검색할 수 있는 검색기 도구입니다. 
        서비스 사용법, 대출 방법, 투자 방법에 대한 질문이 들어오면 사용합니다.
        예시
        - 나 대출 받는 방법이 궁금해.
        - 투자를 하고싶은데 어떻게 해.
        - Billit 어떻게 써?
        - 이 서비스는 뭐하는 서비스야?
        이런 질문에는 이 local retriever를 써 관련된 문서를 찾은 후 그 문서를 기반으로 답하세요.'''
    )
    return retriever_tool

def fin_retirever():
    retriever_tool = create_retriever_tool(
        agent_retirever2,
        name = 'financial_vocabulary',
        description = '''금융지식에 관한 문서들을 검색할 수 있는 검색기 도구입니다.
        금융지식 관련 질문이 들어오면 사용합니다.
        예시
        - 자료열람요구권이 뭐야?
        - 개인신용평가대응권이 뭐야?
        - 인지세가 뭐야?
        - 개인신용평가대응권을 못쓰는 상황이 어떻게 돼?
        - 인지세는 누가, 얼마나 부담해?
    '''
    )
    return retriever_tool