import funcsfortool
from funcsfortool import SimpleScreening, retrieve
from langchain.agents import AgentExecutor, create_react_agent, Tool, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain_core.runnables.utils import ConfigurableFieldSpec
import time
from langchain_core.runnables.history import RunnableWithMessageHistory

simple = SimpleScreening()
retriever_tool = retrieve()
template = """
Important를 지키면서 다음 질문에 답변하세요. 다음 도구들을 사용할 수 있습니다:

{tools}

다음 형식을 사용하세요:

Question: 답변해야 할 입력 질문
History: 저장된 대화 내역을 참고하세요.:
{chat_history}
Thought: 무엇을 해야 할지 항상 생각하세요
Action: 취할 행동을 선택하세요. [{tool_names}] 중 하나를 꼭 선택해야 합니다.
Action Input: 행동에 필요한 입력값
Observation: 행동의 결과
... (이 생각/행동/행동 입력/관찰 과정을 N번 반복할 수 있음)
Thought: 필요한 정보를 얻었으므로 최종 답변을 생성하겠습니다
Final Answer: 원래 입력 질문에 대한 최종 답변

Important :
- '그 둘의 차이가 뭐야?', '내가 물어봤던거 다시 알려줘.', '그거 다시 설명해줘.'등의 질문에는 반드시 History를 기반으로 답하세요.
- simple_screening의 Action Input은 simple_screening.args와 simple_screening.description에 맞게 값을 찾아서 채워 넣으세요.
- 'simple_screening'에 누락된 파라미터가 있을 경우 다시 요청하세요. 반드시 다시 요청해야 합니다. 다 채워질 때 까지 계속 요청하세요.
- 주어진 질문에 대한 대답만 하세요. 더 추가적인 정보를 생성하지 마세요.
- 질문 언어와 같은 언어로 답변하세요. 만약 질문이 베트남어이면 베트남어로 답하면 됩니다. 질문이 한국어면 한국어로 답하세요.
- 질문에 대한 정확한 답을 생성하려고 3번 이상 반복하지 마세요.
- Action을 뭘 할지 모르겠다면 local_retriever를 쓰세요.


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
        self.tools = [simple, retriever_tool]
        self.memory = SQLChatMessageHistory(
            table_name = self.user_id,
            session_id = self.session_id,
            connection = self.engine
        )
        self.agent = self.get_agent()
        self.agent_hist = self.agent_history()
    
    def get_agent(self):
        using_tools = self.tools
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
        print(f"Query being passed to the agent: {query}")
        result = agent.invoke({'input' : query}, config = {'configurable' : {'session_id' : self.session_id}})
        print("result :", result)
        memory.add_user_message(query)
        memory.add_ai_message(result['output'])
        return result

template2 = """
Important를 지키면서 다음 질문에 답변하세요. 다음 도구들을 사용할 수 있습니다:

{tools}

다음 형식을 사용하세요:

Question: 답변해야 할 입력 질문
Thought: 무엇을 해야 할지 항상 생각하세요
Action: 취할 행동을 선택하세요. [{tool_names}] 중 하나를 꼭 선택해야 합니다.
Action Input: 행동에 필요한 입력값
Observation: 행동의 결과
... (이 생각/행동/행동 입력/관찰 과정을 N번 반복할 수 있음)
Thought: 필요한 정보를 얻었으므로 최종 답변을 생성하겠습니다
Final Answer: 원래 입력 질문에 대한 최종 답변

Important :
- 주어진 질문에 대한 대답만 하세요. 더 추가적인 정보를 생성하지 마세요.
- 질문 언어와 같은 언어로 답변하세요. 만약 질문이 베트남어이면 베트남어로 답하면 됩니다. 질문이 한국어면 한국어로 답하세요.
- 질문에 대한 정확한 답을 생성하려고 3번 이상 반복하지 마세요.


시작하세요!


Question: {input}
Thought: {agent_scratchpad}
"""

class NonLoginAgent(LoginAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.tools =  [
            Tool(
                name = funcsfortool.retrieve().name,
                func = funcsfortool.retrieve().func,
                description = funcsfortool.retrieve().description)
        ]
        self.prompt = ChatPromptTemplate.from_template(template2)
        self.agent = self.get_agent()
    
    def answer_to_me(self, query):
        agent = self.agent
        result = agent.invoke({'input' : query})
        return result
    
    
