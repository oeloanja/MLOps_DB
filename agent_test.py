from agent import LoginAgent
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import openai

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api
db = "mysql+pymysql://root:1234@localhost:3306/chat_history"
user_id = 'sawadikap2974671'

llm_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=2048,
    model_name="gpt-4o-mini"
)


test_obj = LoginAgent(llm_model, db_path = db, user_id = user_id)
# print(test_obj.prompt)
# print('----------')
# print(test_obj.get_tools())
# print('-----------')
test_agent = test_obj.get_agent()
# print(test_agent.tools)
# print('-----------')
# print(test_agent.get_prompts())

# start1 = time.time()
# result1 = test_agent.invoke({'input':'개인신용평가대응권이 뭐야?'}, config = {"configurable" : {"session_id" : test_obj.get_conversation_id()}})
# end1 = time.time()
# time1 = end1 - start1
# print(result1)
# print(time1)

# start2 = time.time()
# result2 = test_agent.invoke({'input':'자료열람요구권이 뭐야?'}, config = {"configurable" : {"session_id" : test_obj.get_conversation_id()}})
# end2 = time.time()
# time2 = end2 - start2
# print(result2)
# print(time2)

# start3 = time.time()
# result3 = test_agent.invoke({'input': '그 둘의 차이가 뭐야?'}, config = {"configurable" : {"session_id" : test_obj.get_conversation_id()}})
# end3 = time.time()
# time3 = end3 - start3
# print(result3)
# print(time3)

start1 = time.time()
result1 = test_obj.answer_to_me('개인신용평가대응권이 뭐야?')['output']
end1 = time.time()
time1 = end1 - start1
print(result1)
print(type(result1))
print(time1)

# start2 = time.time()
# result2 = test_obj.answer_to_me('자료열람요구권이 뭐야?')['output']
# end2 = time.time()
# time2 = end2 - start2
# print(result2)
# print(time2)

# start3 = time.time()
# result3 = test_obj.answer_to_me('그 둘의 차이가 뭐야?')['output']
# end3 = time.time()
# time3 = end3 - start3
# print(result3)
# print(time3)

# start4 = time.time()
# result4 = test_obj.answer_to_me('나 대출 가능해? 연봉은 2000이야.')
# end4 = time.time()
# time4 = end4 - start4
# print(result4)
# print(time4)