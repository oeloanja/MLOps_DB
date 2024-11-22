from Retriever import retriever
import VectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
#from messagehistory import MessageHistory
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


system_prompt = "당신은 한국에 사는 외국인 노동자들을 위한 대출 상담 챗봇입니다. question과 맞는 언어로 답해야 합니다. 만약 question이 태국어로 되어있으면 태국어로 답해야 합니다. question의 언어를 모르는 경우 한국어로 답하세요. 모르는 부분은 retriever를 꼭 활용하세요. 토큰을 다 안써도 되니 답변을 짧고 간결하게 생성하세요. 다음의 예시를 참고해 question에 맞는 답변을 생성하세요. 예시 - 질문:강아지가 뭐야? 답변:강아지는 어린 개를 의미합니다."

class NonLoginChatbotChain():
    def __init__(self, model, tokenizer, session_id:str, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.session_id = session_id
        self.store = {}
        self.prompt = ChatPromptTemplate.from_messages([
                                            ("system", system_prompt),
                                            ("human", '{input}'),
                                            ("assistant", "{retriever}")
                                        ])
        

    def get_session_history(self) -> BaseChatMessageHistory:
        session_id = self.session_id
        store = self.store
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    
    
    def _get_llmchain(self):
        tokenizer = self.tokenizer
        pad_token = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        eos_token = tokenizer.convert_tokens_to_ids("<eot_id>")
        gen_pipeline = pipeline(model = self.model, tokenizer = tokenizer, task = 'text-generation', return_full_text = False, max_new_tokens = 256, pad_token_id = pad_token, eos_token_id = eos_token)
        llm = HuggingFacePipeline(pipeline = gen_pipeline)
        prompt = self.prompt
        llm_chain = prompt | llm
        return llm_chain
    
    def get_chain_with_rag(self, dir_path, collection, k):
        vec_db = VectorStore.load_vectorstore(dir_path, collection)
        chain_retriever = retriever(vec_db, searched = k).get_retriever()
        llm_chain = self._get_llmchain()
        session_history = self.get_session_history().messages
        rag_history = dict(zip(session_history))
        rag_context = {"input" : RunnablePassthrough(), "retriever" : chain_retriever, "history" : rag_history}
        rag_chain = rag_context | llm_chain | StrOutputParser()
        rag_chain_with_history = RunnableWithMessageHistory(itemgetter('input') | rag_chain, self.get_session_history,
                                                            input_messages_key = "input",
                                                            history_messages_key = "history")
        return rag_chain_with_history
'''    
class LoginChatbotChain():
    def __init__(self, connectstring, table, model, tokenizer, *args, **kwargs):
        self.history_obj = MessageHistory(connectstring = connectstring, table = table)
        self.model = model
        self.tokenizer = tokenizer
'''    
    
