from transformers import AutoTokenizer, AutoModelForCausalLM
from chatbot import NonLoginChatbotChain
from Retriever import retriever
import VectorStore
import time


model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
model_id2 = "NakJun/Llama-3.2-1B-Instruct-ko-QuAD"
llm_model = AutoModelForCausalLM.from_pretrained(model_id2, cache_dir = './LLM2')
llm_tokenizer = AutoTokenizer.from_pretrained(model_id2, cache_dir = './TOKENIZER2')


test_chatbot_chain = NonLoginChatbotChain(model = llm_model, tokenizer = llm_tokenizer, session_id = 'test')
rag_chatbot = test_chatbot_chain.get_chain_with_rag(dir_path='./MLOps_chatbot', collection='testdb', k = 3)
start_time1 = time.time()
result = rag_chatbot.invoke({"input" : "개인신용평가대응권이 뭐야?"}, config = {"configurable" : {"session_id" : 'test'}})
end_time1 = time.time()
time_range1 = end_time1 - start_time1
start_time2 = time.time()
result2 = rag_chatbot.invoke({"input" : "자료열람요구권이 뭐야?"}, config = {"configurable" : {"session_id" : 'test'}})
end_time2 = time.time()
time_range2 = end_time2 - start_time2
start_time3 = time.time()
result3 = rag_chatbot.invoke({"input" : "처음에 물어본 내용에 대해 다시 알려줘"}, config = {"configurable" : {"session_id" : 'test'}})
end_time3 = time.time()
time_range3 = end_time3 - start_time3
print(result)
print(f'실행 시간1 : {time_range1:.3f}')
print(result2)
print(f'실행시간2 : {time_range2:.3f}')
print(result3)
print(f'실행시간3 : {time_range3:.3f}')
print(test_chatbot_chain.get_session_history().messages)
history = test_chatbot_chain.get_session_history().messages
print(f'실행시간 총합 : {time_range1:.3f} + {time_range2:.3f}')
