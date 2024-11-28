from fastapi import FastAPI, APIRouter
from flask import Flask, request
from chatbotchain import NonLoginChain, LoginChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import getuserid
import jsonify


app = Flask(__name__)

model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
llm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = './LLM', do_sample = False)
llm_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = './TOKENIZER')
db = "mysql+pymysql://root:1234@localhost:3306/chat_history"
dir_path = './MLOps_chatbot'
collection = 'testdb'
k = 3
user_id = 'nenguyen2002'

chain_non = NonLoginChain(dir_path, collection, k, llm_model, llm_tokenizer)
chain_login = LoginChain(llm_model, llm_tokenizer, user_id, db, dir_path, collection, k)

# @app.route('/borrow', methods = ['GET'])
# def get_borrow_id():
#     return getuserid.GetUserIDBorrow()

# @app.route('/invest', methods = ['GET'])
# def get_invest_id():
#     return getuserid.GetUserIDInvest()

# def get_id():
#     user_id = get_borrow_id()
#     if user_id is None:
#         user_id = get_invest_id()
#     else :
#         return None
#     return user_id


def chatbot_back(message):
    if user_id is None:
        result = chain_non.answer_to_me({'input' : message})
    else:
        result = chain_login.answer_to_me({'input' : message})
    return result

@app.route('/chat', methods = ['POST'])
def chat():
    question = request.json.get('input')
    response = chatbot_back(question)
    return jsonify({'response' : response})

    

if __name__ == "__main__":
    app.run(host='0.0.0.0')