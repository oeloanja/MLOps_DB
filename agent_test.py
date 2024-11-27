from agent import Agent
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from chatbotchain import LoginChain
import pickle

model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
llm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = './LLM', do_sample = False)
llm_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = './TOKENIZER')
db = "mysql+pymysql://root:1234@localhost:3306/chat_history"
user_id = 'nguyen777'
dir_path = './MLOps_chatbot'
collection = 'testdb'
chain_obj = LoginChain(llm_model, llm_tokenizer, user_id, db, dir_path, collection, 3)
test_chain = chain_obj.get_rag_chain_history()
model = pickle.load(open('tool_ml.pickle', 'rb'))

test_obj = Agent(test_chain, model)
test_agent = test_obj.get_agent()

start1 = time.time()
result1 = test_agent.run('개인신용평가대응권이 뭐야?')
end1 = time.time()
time1 = end1 - start1
print(result1)
print(time1)

start2 = time.time()
result2 = test_agent.run('자료열람요구권이 뭐야?')
end2 = time.time()
time2 = end2 - start2
print(result2)
print(time2)

start3 = time.time()
result3 = test_agent.run('그 둘의 차이가 뭐야?')
end3 = time.time()
time3 = end3 - start3
print(result3)
print(time3)