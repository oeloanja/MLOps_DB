from Retriever import retriever
import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from Store_Messages import store_message


class chatbot_chain():
    def __init__(self, model, tokenizer, user_id, retriever, db_connection, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.user_id = user_id
        self.retriever = retriever
        self.db_connection = db_connection
        self.prompt = ChatPromptTemplate([

                                            ("system", "당신은 한국에 사는 외국인 노동자들을 위한 대출 상담 챗봇입니다. 사용자가 외국어로 물어봤을 때 해당 언어로 답해야 합니다"),
                                            ("human", "{user_input}")
                                            ("Context", "{context}")
                                            ("ai", "")
                                        ])

    def get_chain(self):
        