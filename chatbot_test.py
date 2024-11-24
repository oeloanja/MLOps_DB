from transformers import AutoTokenizer, AutoModelForCausalLM
from chatbot import NonLoginChatbotChain
from Retriever import retriever
import VectorStore
import time
from getretrieverchain import RagChain

model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
model_id2 = "NakJun/Llama-3.2-1B-Instruct-ko-QuAD"
llm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = './LLM')
llm_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = './TOKENIZER')

'''
test_chatbot_chain = NonLoginChatbotChain(model = llm_model, tokenizer = llm_tokenizer, session_id = 'test')
rag_chatbot = test_chatbot_chain.get_chain_with_rag(dir_path='./MLOps_chatbot', collection='testdb', k = 3)
'''
test2 = RagChain(dir_path='./MLops_chatbot', collection='testdb', searched=3, llm=llm_model, tokenizer=llm_tokenizer)
rag_chatbot = test2.get_rag_chain()
start_time1 = time.time()
result = test2.answer_to_me("개인신용평가대응권이 뭐야?")
end_time1 = time.time()
time_range1 = end_time1 - start_time1
start_time2 = time.time()
result2 = test2.answer_to_me("자료열람요구권이 뭐야?")
end_time2 = time.time()
time_range2 = end_time2 - start_time2
start3 = time.time()
result3 = test2.answer_to_me("처음 물어본 질문이 뭐였지?")
end3 = time.time()
time_range3 = end3 - start3
print(result)
print(f'실행 시간1 : {time_range1:.3f}')
print(result2)
print(f'실행시간2 : {time_range2:.3f}')
print(result3)
print(f'실행시간 : {time_range3:.3f}')
print(f'실행시간 총합 : {time_range1:.3f} + {time_range2:.3f}')
