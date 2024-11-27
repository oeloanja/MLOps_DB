from langchain.tools import tool


@tool
def get_simple_screening(ml, income:float, job_duration:int, dti:float, loan_amnt:float):
    """주어진 머신러닝 모델을 이용해 간단한 대출심사를 진행."""
    return ml.predict(loan_amnt, dti, job_duration, income)