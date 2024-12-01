from agent import Agent
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
# from langchain_ollama import ChatOllama
# from llmchain import GetChain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import openai

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api

llm_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=2048,
    model_name="gpt-4o-mini"
)


test_obj = Agent(llm_model)
test_agent = test_obj.get_agent()

start1 = time.time()
result1 = test_agent.invoke({'input':'개인신용평가대응권이 뭐야?'})
end1 = time.time()
time1 = end1 - start1
print(result1)
print(time1)

start2 = time.time()
result2 = test_agent.invoke({'input':'자료열람요구권이 뭐야?'})
end2 = time.time()
time2 = end2 - start2
print(result2)
print(time2)

# start3 = time.time()
# result3 = test_agent.invoke('그 둘의 차이가 뭐야?')
# end3 = time.time()
# time3 = end3 - start3
# print(result3)
# print(time3)