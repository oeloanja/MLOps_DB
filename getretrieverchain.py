from Retriever import retriever
import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from langchain.schema import HumanMessage, AIMessage


system_prompt = """
                당신은 외국인 노동자들을 위한 금융 상담 챗봇입니다.
                질문에 맞는 언어로 답변을 생성해야 합니다.
                질문이 태국어인 경우 태국어로 답해야 합니다.
                질문의 언어를 모르는 경우 한국어로 답해야 합니다.
                답을 생성할 땐 무조건 짧고 간결하게 해야합니다.
                또한 메세지 기록이 있는 경우 그 기록도 참고해서 답변을 생성해야 합니다.
                저번에 한 질문이 뭐였지? 라는 질문엔 저번에 한 질문만 찾아서 답해주면 됩니다.
                """

class RagChain():
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
        gen_pipeline = pipeline(model = model, tokenizer = tokenizer, task = 'text-generation', return_full_text = False, max_new_tokens = 256, pad_token_id = pad_token, eos_token_id = eos_token)
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
        