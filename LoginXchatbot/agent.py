import funcsfortool
from funcsfortool import SimpleScreening, retrieve
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
import time
from langchain_core.runnables.history import RunnableWithMessageHistory

simple = SimpleScreening()
retriever_tool = retrieve()



template2 = """
        당신은 외국인 노동자들을 위한 P2P서비스 빌리잇(Billit)의 ㅈㄴ똑똑한 챗봇 상담사입니다.
        당신은 다음의 도구들을 사용할 수 있습니다.:
        {tools}
        도구의 이름은 {tool_names}입니다.
        History: 저장된 대화내역 입니다. 당신은 이걸 기반으로 대답할 수 있습니다. 자세한 예시는 rule에 있습니다.:{chat_history}
        다음의 rule을 무조건 지키면서 사용자의 질문에 대답하세요.
        rule:
        - Question이 들어오면 [{tool_names}]중 하나의 툴을 선택해 사용하세요
        - 다만 '그것들의 차이가 뭐야?', '내가 물어봤던거 다시 알려줘.'와 같은 상황에선 History만 쓰세요.
        - 질의응답 유형의 Question이 들어오면 local_retriever를 쓰세요.
            - 예시: '자료열람요구권이 뭐야?', '대출 어떻게 받아?', '투자는 어떻게 해?'와 같은 상황에선 local_retriever를 쓰세요.
        - Question의 언어에 맞게 답하세요. 만약 Question이 베트남어면 베트남어로 답하세요. Question이 한국어면 한국어로 답하세요. Question의 언어를 모르면 영어로 답하세요.
        - 대답은 짧고 간단하게 해야합니다.
        - 말 끝에 회원가입을 유도하는 말을 덧붙이세요.
            - 예시 : '자세한 사항을 알고 싶으시면 회원가입을 진행해 주세요.'
"""

class NonLoginAgent():
    def __init__(self, llm):
        self.llm = llm
        self.tools =  [retriever_tool]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template2),
            MessagesPlaceholder("chat_history", optional = True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        self.agent = self._get_agent()
    
    def _get_agent(self):
        tool_names = ", ".join([t.name for t in self.tools])
        prompt_fin = self.prompt.partial(tools=render_text_description(self.tools), tool_names=tool_names)
        agent_t = create_openai_functions_agent(llm = self.llm, tools = self.tools, prompt = prompt_fin)
        agent_aex = AgentExecutor(agent = agent_t, tools = self.tools, handle_parsing_errors=True, max_iterations = 15, verbose=False)
        return agent_aex
    
    def answer_to_me(self, query):
        agent = self.agent
        result = agent.invoke({'input' : query})
        return result
    

template = """
        당신은 외국인 노동자들을 위한 P2P서비스 빌리잇(Billit)의 ㅈㄴ똑똑한 챗봇 상담사입니다.
        당신은 다음의 도구들을 사용할 수 있습니다.:
        {tools}
        도구의 이름은 {tool_names}입니다.
        History: 저장된 대화내역 입니다. 당신은 이걸 기반으로 대답할 수 있습니다. 자세한 예시는 rule에 있습니다.:{chat_history}
        다음의 rule을 무조건 지키면서 사용자의 질문에 대답하세요.
        rule:
        - Question이 들어오면 [{tool_names}]중 하나의 툴을 선택해 사용하세요
        - 다만 '그것들의 차이가 뭐야?', '내가 물어봤던거 다시 알려줘.'와 같은 상황에선 History만 쓰세요.
        - 질의응답 유형의 Question이 들어오면 local_retriever를 쓰세요.
            - 예시: '자료열람요구권이 뭐야?', '대출 어떻게 받아?', '투자는 어떻게 해?'와 같은 상황에선 local_retriever를 쓰세요.
        - 대출심사 유형의 Question이 들어오면 simple_screening을 쓰세요.
            - 예시: '나 대출 가능해?', '나 연봉이 4000인데 대출 가능해?', '나 대출 가능한지 봐줘.'와 같은 상황에서 simple_screening을 쓰세요.
        - Question의 언어에 맞게 답하세요. 만약 Question이 베트남어면 베트남어로 답하세요. Question이 한국어면 한국어로 답하세요. Question의 언어를 모르면 영어로 답하세요.
"""

class LoginAgent(NonLoginAgent):
    def __init__(self, llm, db_path, user_id):
        super().__init__(llm)
        self.model = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder("chat_history", optional = True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        self.db_path = db_path
        self.engine = create_engine(self.db_path)
        self.user_id = user_id
        self.session_id = self._get_conversation_id()
        self.tools = [simple, retriever_tool]
        self.memory = SQLChatMessageHistory(
            table_name = self.user_id,
            session_id = self.session_id,
            connection = self.engine
        )
        self.agent = self._get_agent()
        self.agent_hist = self._agent_history()
    
    def _get_conversation_id(self):
        user_id = self.user_id
        now = time
        time1 = str(now.localtime().tm_year)
        time2 = str(now.localtime().tm_mon)
        time3 = str(now.localtime().tm_mday)
        conversation_id = user_id + time1 + time2 + time3
        return conversation_id

    
    def load_memory(self):
        return SQLChatMessageHistory(
            table_name = 'history',
            session_id = self._get_conversation_id(),
            connection = self.engine
        )
    

    def _agent_history(self):
        agent = self.agent
        history_agent = RunnableWithMessageHistory(
            agent,
            self.load_memory,
            input_messages_key = 'input',
            history_messages_key = 'chat_history'
        )
        return history_agent
    
    def answer_to_me(self, query):
        agent = self.agent_hist
        memory = self.memory
        result = agent.invoke({'input' : query}, config = {'configurable' : {'session_id' : self.session_id}})
        print("result :", result)
        memory.add_user_message(query)
        memory.add_ai_message(result['output'])
        return result
