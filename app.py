from fastapi import FastAPI, APIRouter
from flask import Flask, request
import getuserid
from langchain_openai import ChatOpenAI
from agent import LoginAgent, NonLoginAgent
from dotenv import load_dotenv
import os


app = Flask(__name__)

load_dotenv()
api = os.getenv('OPENAPI')
os.environ['OPENAI_API_KEY'] = api

llm_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=2048,
    model_name="gpt-4o-mini"
)
db = "mysql+pymysql://root:1234@localhost:3306/chat_history"



@app.route('/borrow', methods = ['GET'])
def get_borrow_id():
    return getuserid.GetUserIDBorrow()

@app.route('/invest', methods = ['GET'])
def get_invest_id():
    return getuserid.GetUserIDInvest()

def get_id():
    user_id = get_borrow_id()
    if user_id is None:
        user_id = get_invest_id()
        if user_id is None:
            return None
    return user_id

user_id = get_id()


login = LoginAgent(llm_model, db_path = db, user_id = user_id)
non = NonLoginAgent(llm_model)


@app.route('/chat/login', methods = ['POST'])
def chat():
    question = request.form['input']
    response = login.answer_to_me(question)
    print(response['output'])
    return response['output']

@app.route('/chat/non', methods = ['POST'])
def chat_non():
    question = request.form['input']
    response = non.answer_to_me(question)
    print(response['output'])
    return response['output']
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8000)