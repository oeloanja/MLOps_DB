import funcsfortool
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description

template = """ 
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
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
                name = 'Retrieve',
                func = funcsfortool.retrieve,
                description = "챗봇 사용자의 질문이 들어오면 그 질문과 관련된 문서를 검색합니다."
            )
        ]
        return tools
    
    def get_agent(self):
        using_tools = self.get_tools()
        agent_t = create_react_agent(llm = self.model, tools = using_tools, prompt = self.prompt_fin)
        agent_aex = AgentExecutor(agent = agent_t, tools = using_tools, handle_parsing_errors=True)
        return agent_aex