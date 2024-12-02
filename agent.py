import funcsfortool
from langchain.agents import AgentExecutor, create_react_agent, Tool, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain_core.runnables.utils import ConfigurableFieldSpec
import time
from langchain_core.runnables.history import RunnableWithMessageHistory



template = """
다음 질문에 답변하세요. 다음 도구들을 사용할 수 있습니다:

{tools}

다음 형식을 사용하세요:

Question: 답변해야 할 입력 질문
History: 저장된 대화 내역을 참고하세요. 무조건 참고해서 대화 맥락에 맞게 답하세요.:
{chat_history}
Thought: 무엇을 해야 할지 항상 생각하세요
Action: 취할 행동을 선택하세요. [{tool_names}] 중 하나를 선택해야 합니다.
Action Input: 행동에 필요한 입력값
Observation: 행동의 결과
... (이 생각/행동/행동 입력/관찰 과정을 N번 반복할 수 있음)
Thought: 필요한 정보를 얻었으므로 최종 답변을 생성하겠습니다
Final Answer: 원래 입력 질문에 대한 최종 답변

중요 :
- 다음의 규칙은 무조건 지켜야 하는 규칙들입니다. 무슨 일이 있어도 지키세요.
- 반드시 History를 참고하세요. 도구들 만으로 답을 하는데 무리가 있을 경우 History를 기반으로 답하세요. 무조건 지켜야 합니다.
- 'simple_screening'에 누락된 파라미터가 있을 경우 다시 요청하세요. 반드시 다시 요청해야 합니다. 다 채워질 때 까지 계속 요청하세요.
- 주어진 질문에 대한 대답만 하세요. 더 추가적인 정보를 생성하지 마세요.
- 질문 언어와 같은 언어로 답변하세요. 만약 질문이 베트남어이면 베트남어로 답하면 됩니다. 질문이 한국어면 한국어로 답하세요.
- 질문에 대한 정확한 답을 생성하려고 3번 이상 반복하지 마세요.
- History를 참고해 문맥을 파악하세요. 반드시 이전 대화 내용을 참고해 답변하세요.
- 무조건  History를 참고해 답변하세요. 이건 무조건 지키세요. 무조건 지켜야 하는 규칙입니다.


시작하세요!


Question: {input}
History: {chat_history}
Thought: {agent_scratchpad}
"""

class LoginAgent():
    def __init__(self, llm, db_path, user_id):
        self.model = llm
        self.prompt = ChatPromptTemplate.from_template(template)
        self.local_retriever = funcsfortool.retrieve()
        self.engine = create_engine(db_path)
        self.user_id = user_id
        self.session_id = self.get_conversation_id()
        self.memory = SQLChatMessageHistory(
            table_name = self.user_id,
            session_id = self.session_id,
            connection = self.engine
        )
        

    def get_tools(self):
        tools = [
            Tool(
                name = 'simple_screening',
                func = funcsfortool.get_simple_screening,
                description = funcsfortool.get_simple_screening.description
            ),
            Tool(
                name = funcsfortool.retrieve().name,
                func = funcsfortool.retrieve().func,
                description = funcsfortool.retrieve().description)
        ]
        return tools
    
    def get_agent(self):
        using_tools = self.get_tools()
        prompt = self.prompt
        tool_names = ", ".join([t.name for t in using_tools])
        prompt_fin = prompt.partial(tools=render_text_description(using_tools), tool_names=tool_names)
        agent_t = create_react_agent(llm = self.model, tools = using_tools, prompt = prompt_fin)
        agent_aex = AgentExecutor(agent = agent_t, tools = using_tools, handle_parsing_errors=True, max_iterations = 15, verbose=True)
        return agent_aex
    
    def get_conversation_id(self):
        user_id = self.user_id
        now = time
        time1 = str(now.localtime().tm_year)
        time2 = str(now.localtime().tm_mon)
        time3 = str(now.localtime().tm_mday)
        conversation_id = user_id + time1 + time2 + time3
        return conversation_id

    
    def load_memory(self):
        return SQLChatMessageHistory(
            table_name = self.user_id,
            session_id = self.get_conversation_id(),
            connection = self.engine
        )
    

    def agent_history(self):
        agent = self.get_agent()
        history_agent = RunnableWithMessageHistory(
            agent,
            self.load_memory,
            input_messages_key = 'input',
            history_messages_key = 'chat_history'
        )
        return history_agent
    
    def answer_to_me(self, query):
        agent = self.agent_history()
        memory = self.memory
        result = agent.invoke({'input' : query}, config = {'configurable' : {'session_id' : self.get_conversation_id()}})
        memory.add_user_message(query)
        memory.add_ai_message(result['output'])
        return result
    

class NonLoginAgent():
    def __init__(self, llm):
        self.llm = llm
    
    def get_tools(self):
        tools = [
            Tool(
                name = funcsfortool.retrieve().name,
                func = funcsfortool.retrieve().func,
                description = funcsfortool.retrieve().description)
        ]
        return tools
