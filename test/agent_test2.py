from agentver2 import LoginAgent
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import openai

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api
db = "mysql+pymysql://root:1234@localhost:3306/chat_history"
user_pn = '010-1818-1818'
user_id = 'F003451231'

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

# start1 = time.time()
# result1 = test_obj.answer_to_me('개인신용평가대응권이 뭐야?')['output']
# end1 = time.time()
# time1 = end1 - start1
# print(result1)
# print(type(result1))
# print(time1)

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
# result4 = test_obj.answer_to_me('나 대출 받고 싶은데 어떻게 받아?')['output']
# end4 = time.time()
# time4 = end4 - start4
# print(result4)
# print(time4)
question = '나 대출 가능해? 연봉은 3000만원이야. 10년 일했고, 5억 대출 받고싶어.'
# if "대출 가능" in question:
#     question = f"{question} (user_pn: 010-1234-5678)"
start5 = time.time()
result5 = test_obj.answer_to_me(question)
end5 = time.time()
time5 = end5 - start5
print(result5)
print(time5)

