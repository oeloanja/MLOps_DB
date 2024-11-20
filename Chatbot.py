from Retriever import retriever
import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from Store_Messages import store_message
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.llms import HuggingFacePipeline

system_prompt = "당신은 한국에 사는 외국인 노동자들을 위한 대출 상담 챗봇입니다. question과 맞는 언어로 답해야 합니다. 만약 question이 태국어로 되어있으면 태국어로 답해야 합니다. question의 언어를 모르는 경우 한국어로 답하세요. 모르는 부분은 retriever를 꼭 활용하세요. 토큰을 다 안써도 되니 답변을 짧고 간결하게 생성하세요. 다음의 예시를 참고해 question에 맞는 답변을 생성하세요. 예시 - 질문:강아지가 뭐야? 답변:강아지는 어린 개를 의미합니다."

class chatbot_chain():
    def __init__(self, model, tokenizer, user_id = None, db_connection = None, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        #self.user_id = user_id
        self.prompt = ChatPromptTemplate([

                                            ("system", system_prompt),
                                            ("human", '{question}'),
                                            ("assistant", "{retriever}")
                                        ])

    
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
        rag_context = {"question" : RunnablePassthrough(), "retriever" : chain_retriever}
        rag_chain = rag_context | llm_chain
        return rag_chain