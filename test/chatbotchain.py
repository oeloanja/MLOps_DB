from Retriever import Retriever
import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from langchain.schema import HumanMessage, AIMessage
#import getuserid
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.agents import initialize_agent, Tool
from langchain_core.tools import tool







system_prompt = """
                당신은 외국인 노동자들을 위한 금융 상담 챗봇입니다.
                질문에 맞는 언어로 답변을 생성해야 합니다.
                질문이 태국어인 경우 태국어로 답해야 합니다.
                질문의 언어를 모르는 경우 한국어로 답해야 합니다.
                답을 생성할 땐 무조건 짧고 간결하게 해야합니다.
                또한 메세지 기록이 있는 경우 그 기록도 참고해서 답변을 생성해야 합니다.
                저번에 한 질문이 뭐였지? 라는 질문엔 저번에 한 질문만 찾아서 답해주면 됩니다.
                """

class NonLoginChain():
    def __init__(self, dir_path, collection, searched, llm):
        self.vec_db = VectorStore.load_vectorstore(dir_path, collection)
        self.retriever = Retriever(self.vec_db, searched)
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("assistant", "{retriever}")
            ]
        )
        self.memory = ConversationBufferMemory(memory_key = 'chat_history', input_key="input", output_key="answer", return_messages=True)

    
    def get_rag_chain(self):
        rag_context = {"chat_history" : RunnableLambda(self.memory.load_memory_variables) | itemgetter(self.memory.memory_key), "input" : RunnablePassthrough(), "retriever" : self.retriever}
        chain = rag_context | self.prompt | self.llm | StrOutputParser()
        return chain
    
    def answer_to_me(self, question):
        chain = self.get_rag_chain()
        memory = self.memory
        result = chain.invoke({"input" : question})
        memory.save_context(
            {"input" : question},
            {"answer" : result}
        )
        return result

template = """
            당신의 역할은 외국인 노동자 전용 P2P 대출 플랫폼인 빌리잇 사용자들을 위한 금융 챗봇입니다.
            사용자의 질문에 맞는 대답을 하세요.
            아래의 규칙을 반드시 따르세요:
            1. 질문에 대한 대답만 생성하세요. 절대 새로운 질문을 생성하지 마세요. 이건 무조건 지켜야 합니다.
            2. 대답은 반드시 짧고 간결하게 하세요. 이건 반드시 지키세요.
            3. 대답 후 추가적인 정보를 제공하거나 다른 주제를 제시하지 마세요.
            4. 사용자 질문 외의 내용은 대답하지 마세요.
            5. 무조건 대화기록과 검색결과를 바탕으로 대답하세요. 이건 무조건 지켜야 합니다.
            6. 답변이 주어진 토큰 수보다 길어져 끊어지는 문제를 막아야 합니다. 그러니 간략하게 요약해서 답변하세요. 메모리 문제로 꼭 요약해야 합니다.

            대화기록 : {chat_history}
            검색결과 : {context}
            질문: {input}
            Answer: 
"""

class LoginChain(NonLoginChain):
    def __init__(self, llm,  user_id, db_url, dir_path, collection, searched):
        super().__init__(dir_path, collection, searched, llm)
        self.model = llm
        self.user_id = user_id
        self.prompt = ChatPromptTemplate.from_template(template)
        self.engine = create_engine(db_url)
        self.memory = SQLChatMessageHistory(session_id = self.user_id, table_name = user_id, connection = self.engine)
        self.vecdb = self.vec_db
        self.retriever = Retriever(self.vecdb, searched)


    def load_memories(self):
        memory = self.memory
        history = memory.get_messages()
        return [msg.pretty_repr() for msg in history]

    def get_rag_chain_history(self):
        print(type(lambda x: self.load_memories()))
        print(type(self.retriever))
        rag_chain = ({"chat_history": lambda x: self.load_memories(),
                      "context" : self.retriever,
                     "input" : itemgetter("input")}
                     | self.prompt
                     | self.llm
                     | StrOutputParser())
        return rag_chain
    
    def answer_to_me(self, question):
        chain = self.get_rag_chain_history()
        result = chain.invoke({"input" : question})
        self.memory.add_user_message(question)
        self.memory.add_ai_message(result)
        return result