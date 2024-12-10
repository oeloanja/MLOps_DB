from flask import Flask, request
from langchain_openai import ChatOpenAI
from agent import LoginAgent, NonLoginAgent
from dotenv import load_dotenv
import os
import requests


app = Flask(__name__)

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api

llm_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=2048,
    model_name="gpt-4o-mini"
)
db = "mysql+pymysql://root:1234@mysqlchatbot/chat_history"

first_flag = False

@app.route('/chat/open', methods = ['POST'])
def get_id():
    global first_flag
    global chatbot
    if not first_flag:
        data = request.get_json()
        response = data['uuid']
        first_flag = True
        chatbot = LoginAgent(llm_model, db_path = db, user_id = response)
    return {"status" : "success"}, 200
    


@app.route('/chat/login', methods = ['POST'])
def chat():
    q_json = request.get_json()
    question = q_json['input']
    response = chatbot.answer_to_me(question)
    out_json = {"output" : response['output']}
    return out_json


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8000)