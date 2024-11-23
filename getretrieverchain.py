from Retriever import retriever
import VectorStore
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

system_prompt = """
                당신은 외국인 노동자들을 위한 금융 상담 챗봇입니다.
                {input}에 맞는 언어로 답변을 생성해야 합니다.
                {input}이 태국어인 경우 태국어로 답해야 합니다.
                {input}의 언어를 모르는 경우 한국어로 답해야 합니다.
                답을 생성할 땐 무조건 짧고 간결하게 해야합니다.
                """

class RagChain():
    def __init__(self, dir_path, collection, searched, llm, tokenizer):
        self.vec_db = VectorStore.load_vectorstore(dir_path, collection)
        self.retriever = retriever(self.vec_db, searched).get_retriever()
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
        self.memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
        
            
    def _get_llm_pipeline(self):
        model = self.llm
        tokenizer = self.tokenizer
        pad_token = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        eos_token = tokenizer.convert_tokens_to_ids("<eot_id>")
        gen_pipeline = pipeline(model = model, tokenizer = tokenizer, task = 'text-generation', return_full_text = False, max_new_tokens = 256, pad_token_id = pad_token, eos_token_id = eos_token)
        llm_pipeline = HuggingFacePipeline(pipeline = gen_pipeline)
        return llm_pipeline
    
    def get_rag_chain(self):
        llm_pipeline = self._get_llm_pipeline()
        rag_context = {"input" : RunnablePassthrough(), "retriever" : self.retriever}
        chain = rag_context | self.prompt | llm_pipeline
        return chain
    
    def load_memory(self):
        chat_history = self.memory.load_memory_variables({})['chat_history']
        if not chat_history:
            return {}
        else:
            return self.memory.load_memory_variables({})['chat_history']
    
    def get_chain_memory(self):
        chain = self.get_rag_chain()
        memory = self.memory
        memory_chain = RunnablePassthrough.assign(chat_history = self.load_memory()) | chain
        return memory_chain
    
    def answer_to_me(self, question):
        chain = self.get_chain_memory()
        memory = self.memory
        result = chain.invoke({"input" : question})
        memory.save_context(
            {"input" : question},
            {"answer" : result.content}
        )
        return result.content
        