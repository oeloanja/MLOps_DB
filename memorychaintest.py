from chatbotchain import LoginChain, NonLoginChain
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
llm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = './LLM', do_sample = False)
llm_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = './TOKENIZER')

db = "mysql+pymysql://root:1234@localhost:3306/chat_history"
user_id = 'nguyen1000'
dir_path = './MLOps_chatbot'
collection = 'testdb'

chain_obj = LoginChain(llm_model, llm_tokenizer, user_id, db, dir_path, collection, 3)
memory_chain = chain_obj.memory_pipeline()
