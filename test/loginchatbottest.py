from chatbotchain import LoginChain, NonLoginChain
from langchain_openai import ChatOpenAI
import time
import os
from dotenv import load_dotenv
import funcsfortool

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api

llm_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=2048,
    model_name="gpt-4o-mini"
)
tools = [funcsfortool.get_simple_screening]
llm = llm_model.bind_tools(tools)


db = "mysql+pymysql://root:1234@localhost:3306/chat_history"
user_id = 'nguyen333'
dir_path = './MLOps_chatbot'
collection = 'testdb'

chain_obj = LoginChain(llm_model, user_id, db, dir_path, collection, 3)
#test_chain = chain_obj.get_rag_chain_history()

start1 = time.time()
result1 = chain_obj.answer_to_me('개인신용평가대응권이 뭐야?')
end1 = time.time()
time1 = end1 - start1
print(result1)
print(time1)
start2 = time.time()
result2 = chain_obj.answer_to_me('자료열람요구권이 뭐야?')
end2 = time.time()
time2 = end2 - start2
print(result2)
print(time2)
start3 = time.time()
result3 = chain_obj.answer_to_me('그 둘의 차이가 뭐야?')
end3 = time.time()
time3 = end3 - start3
print(result3)
print(time3)