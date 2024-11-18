from transformers import AutoTokenizer, AutoModelForCausalLM
from Chatbot import chatbot_chain
from Retriever import retriever
import VectorStore


model_id = "allganize/Llama-3-Alpha-Ko-8B-Evo"
llm_model = AutoModelForCausalLM.from_pretrained(model_id)
llm_tokenizer = AutoTokenizer.from_pretrained(model_id)


test_chatbot_chain = chatbot_chain(model = llm_model, tokenizer = llm_tokenizer, user_id = 'abc123')
rag_chatbot = test_chatbot_chain.get_chain_with_rag(dir_path='./MLOps_chatbot', collection='testdb', k = 3)
result = rag_chatbot.invoke("마이너스통장이 뭐야?")
print(result)

