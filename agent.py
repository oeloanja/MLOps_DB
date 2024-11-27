import predicttool
from langchain.agents import AgentExecutor, create_react_agent, Tool



class Agent():
    def __init__(self, chain, ml):
        self.chain_obj = chain
        self.ml = ml
        
    def get_tools(self):
        tools = [
            Tool(
                name = 'Simple Screening',
                func =lambda **kwargs: predicttool.get_simple_screening(self.ml, **kwargs),
                description = "챗봇 사용자에게 간단한 대출심사를 해줍니다. '나 대출이 가능한지 궁금해', '내가 대출이 가능해?', '간이 대출심사 받고싶어'등의 요청이 들어오면 이 tool을 사용해 답변을 생성합니다. 필요한 파라미터는 모두 받아야 합니다."
            )
        ]
        return tools
    
    def get_agent(self):
        using_tools = self.get_tools()
        agent_t = create_react_agent(tools = using_tools, llm = self.chain_obj)
        agent_aex = AgentExecutor(agent = agent_t, tools = using_tools)
        return agent_aex