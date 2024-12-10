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




dirpath = './MLOps_chatbot'
collection = 'testdb'
vec_db = VectorStore.load_vectorstore(dir_path = dirpath, collection_name = collection)
ret = Retriever(dirpath, collection, searched = 2)
agent_retirever = ret.as_retriever()


ml = pickle.load(open('tool_ml.pickle', 'rb'))

class ScreeningInput(BaseModel):
   annual_income:int = Field(..., description = "사용자가 입력한 질문에 있는 연봉. 혹은 사용자가 입력한 질문에 있는 소득.")
   career_years:int = Field(..., description = "사용자가 입력한 질문에 있는 경력. 혹은 사용자가 입력한 질문에 있는 근속년수.")
   loan_amount:int = Field(..., description = "사용자가 입력한 질문에 있는 대출 희망 금액.")
   user_id:str = Field(description = 'dti 계산을 위한 데이터를 불러오는 키. 툴을 사용할 땐 챗봇이 알아서 전달 해줘야하는 파라미터.')

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
    user_id는 self.user_id입니다. self.user_id를 받으면 됩니다.
    TypeError시 사용자에게 다시 입력 받으세요.
    """
    args_schema : Type[BaseModel] = ScreeningInput
    def _run(self, annual_income:int, career_years:int, loan_amount:int, user_id:str, run_manager: Optional[CallbackManagerForToolRun] = None):
        if career_years > 10:
            career_years = 10
        dti = getdti.calculate_dti(user_id, annual_income)
        input_data = pd.DataFrame([{'loan_amnt' : loan_amount, 'dti' : dti, 'job_duration' : career_years, 'annual_inc' : annual_income}])
        prediction = ml.predict(input_data)
        screening_result = ''
        if prediction[0] == 0:
            screening_result = '대출이 가능합니다.'
        elif prediction[0] == 1:
            screening_result = '대출이 가능하지만 금리가 조금 더 높을 수 있어요.'
        elif prediction[0] == 2:
            screening_result = '대출이 되더라도 금리가 많이 높아요.'
        else:
            screening_result = '대출 안돼. 안바꿔줘. 돌아가.'
        return screening_result

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