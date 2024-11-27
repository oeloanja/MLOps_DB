from Retriever import retriever
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
    def __init__(self, dir_path, collection, searched, llm, tokenizer):
        self.vec_db = VectorStore.load_vectorstore(dir_path, collection)
        self.retriever = retriever(self.vec_db, searched)
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("assistant", "{retriever}")
            ]
        )
        self.memory = ConversationBufferMemory(memory_key = 'chat_history', input_key="input", output_key="answer", return_messages=True)

    def _get_llm_pipeline(self):
        model = self.llm
        tokenizer = self.tokenizer
        pad_token = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        eos_token = tokenizer.convert_tokens_to_ids("<eot_id>")
        gen_pipeline = pipeline(model = model, tokenizer = tokenizer, task = 'text-generation', return_full_text = False, max_new_tokens = 128, pad_token_id = pad_token, eos_token_id = eos_token)
        llm_pipeline = HuggingFacePipeline(pipeline = gen_pipeline)
        return llm_pipeline
    
    def get_rag_chain(self):
        llm_pipeline = self._get_llm_pipeline()
        rag_context = {"chat_history" : RunnableLambda(self.memory.load_memory_variables) | itemgetter(self.memory.memory_key), "input" : RunnablePassthrough(), "retriever" : self.retriever}
        chain = rag_context | self.prompt | llm_pipeline
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
            당신은 카페인, 니코틴 중독 개발자 전영욱이 만든 챗봇입니다.
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
    def __init__(self, llm, tokenizer, user_id, db_url, dir_path, collection, searched):
        super().__init__(dir_path, collection, searched, llm, tokenizer)
        self.model = llm
        self.tokenizer = tokenizer
        #self.ml = ml
        self.user_id = user_id
        self.prompt = ChatPromptTemplate.from_template(template)
        self.engine = create_engine(db_url)
        self.memory = SQLChatMessageHistory(session_id = self.user_id, table_name = user_id, connection = self.engine)
        self.vecdb = self.vec_db
        self.retriever = retriever(self.vecdb, searched)

    # def get_simple_screening(self, income:float, job_duration:int, dti:float, loan_amnt:float):
    #     """주어진 머신러닝 모델을 이용해 간단한 대출심사를 진행."""
    #     ml = self.ml
    #     return ml.predict(loan_amnt, dti, job_duration, income)
    
    # def get_tools(self):
    #     simple_screening_tools = {
    #         "type" : "function",
    #         "function" : {
    #             "name" : "get_simple_screening",
    #             "description" : "고객에 대한 간단한 대출심사를 합니다. 예를들어 '나 대출심사 해줘'라는 요청이 오면 파라미터를 입력 받아 정의된 함수를 호출해 결과를 알려줍니다.",
    #             "parameters" : {
    #                 "income" : {
    #                     "type" : float,
    #                     "description" : "연봉"
    #                 },
    #                 "job_duration" : {
    #                     "type" : int,
    #                     "description" : "경력"
    #                 },
    #                 "dti" : {
    #                     "type" : float,
    #                     "description" : "소득 대비 부채 비율"
    #                 },
    #                 "loan_amnt" : {
    #                     "type" : float,
    #                     "description" : "대출 받고자 하는 금액"
    #                 },
    #                 "required" : ["income", "job_duration", "dti", "loan_amnt"],
    #                 "additionalProperties" : False
    #             }
    #         }
    #     }
    #     tools = [simple_screening_tools]
    #     return tools


    def load_memories(self):
        memory = self.memory
        history = memory.get_messages()
        return [msg.pretty_repr() for msg in history]

    def get_rag_chain_history(self):
        llm_pipe = self._get_llm_pipeline()
        rag_chain = ({"chat_history": lambda x: self.load_memories(),
                      "context" : self.retriever,
                     "input" : itemgetter("input")}
                     | self.prompt
                     | llm_pipe
                     | StrOutputParser())
        return rag_chain
    
    def answer_to_me(self, question):
        chain = self.get_rag_chain_history()
        result = chain.invoke({"input" : question})
        self.memory.add_user_message(question)
        self.memory.add_ai_message(result)
        return result