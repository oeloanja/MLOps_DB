from fastapi import FastAPI, APIRouter
from flask import Flask, request, session, g, Blueprint
from langchain_openai import ChatOpenAI
from agent import NonLoginAgent
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

chatbot = NonLoginAgent(llm_model)

@app.route('/chat/non', methods = ['POST'])
def chat_non():
    q_json = request.get_json()
    question = q_json['input']
    response = chatbot.answer_to_me(question)
    out_json = {"output" : response['output']}
    return out_json

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8070)