import funcsfortool
from langchain.agents import AgentExecutor, create_react_agent, Tool, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description


template = """
다음 질문에 최대한 정확하게 답변하세요. 다음 도구들을 사용할 수 있습니다:

{tools}

다음 형식을 사용하세요:

질문: 답변해야 할 입력 질문
생각: 무엇을 해야 할지 항상 생각하세요
행동: 취할 행동을 선택하세요. [{tool_names}] 중 하나를 선택해야 합니다.
행동 입력: 행동에 필요한 입력값
관찰: 행동의 결과
... (이 생각/행동/행동 입력/관찰 과정을 N번 반복할 수 있음)
생각: 필요한 정보를 얻었으므로 최종 답변을 생성하겠습니다
최종 답변: 원래 입력 질문에 대한 최종 답변


- 주어진 질문에 대한 대답만 하세요. 더 추가적인 정보를 생성하지 마세요.
- 질문 언어와 같은 언어로 답변하세요. 만약 질문이 베트남어이면 베트남어로 답하면 됩니다. 질문이 한국어면 한국어로 답하세요.
- 최종 답변이 생성되면 바로 사용자에게 답하세요.

시작하세요!

질문: {input}
생각: {agent_scratchpad}
"""



class Agent():
    def __init__(self, llm):
        self.model = llm
        self.prompt = ChatPromptTemplate.from_template(template)
        self.local_retriever = funcsfortool.retrieve()
        
    def get_tools(self):
        tools = [
            Tool(
                name = funcsfortool.get_simple_screening.name,
                func = funcsfortool.get_simple_screening,
                description = funcsfortool.get_simple_screening.description
            ),
            Tool(
                name = funcsfortool.retrieve().name,
                func = funcsfortool.retrieve,
                description = funcsfortool.retrieve().description)
        ]
        return tools
    
    def get_agent(self):
        prompt_fin = self.prompt.partial(tools=render_text_description(self.get_tools()), tool_names=", ".format(t.name for t in self.get_tools()))
        using_tools = self.get_tools()
        agent_t = create_structured_chat_agent(llm = self.model, tools = using_tools, prompt = prompt_fin)
        agent_aex = AgentExecutor(agent = agent_t, tools = using_tools, handle_parsing_errors=True, max_iterations = 15, verbose=True)
        return agent_aex