import funcsfortool
from langchain.agents import AgentExecutor, create_react_agent, Tool
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
생각: 이제 최종 답을 알겠습니다.
최종 답변: 원래 입력 질문에 대한 최종 답변

중요:
- `Simple Screening`은 대출 심사 요청이 있을 때만 사용합니다. 그 외의 작업에서는 절대 쓰지 마세요.
- `Retriever`는 항상 사용해야 하는 도구입니다.
- 어떤 tool을 써야할 지 모를 땐 Retriever를 쓰세요.

시작하세요!

질문: {input}
생각: {agent_scratchpad}
"""

class Agent():
    def __init__(self, llm):
        self.model = llm
        self.prompt = ChatPromptTemplate.from_template(template)
        self.prompt_fin = self.prompt.partial(tools=render_text_description(self.get_tools()), tool_names=", ".format(t.name for t in self.get_tools()))
        
    def get_tools(self):
        tools = [
            Tool(
                name = 'Simple Screening',
                func = funcsfortool.get_simple_screening,
                description = "챗봇 사용자에게 간단한 대출심사를 해줍니다. '나 대출이 가능한지 궁금해', '내가 대출이 가능해?', '간이 대출심사 받고싶어'등의 심사 요청이 들어왔을 때만 이 tool을 사용해 답변을 생성합니다."
            ),
            Tool(
                name = 'Retriever',
                func = funcsfortool.retrieve,
                description = "챗봇 사용자의 질문이 들어오면 그 질문과 관련된 문서를 검색합니다."
            )
        ]
        return tools
    
    def get_agent(self):
        using_tools = self.get_tools()
        agent_t = create_react_agent(llm = self.model, tools = using_tools, prompt = self.prompt)
        agent_aex = AgentExecutor(agent = agent_t, tools = using_tools, handle_parsing_errors=True, max_iterations = 15)
        return agent_aex