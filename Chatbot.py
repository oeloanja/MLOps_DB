from Retriever import retriever
import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from Store_Messages import store_message
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.llms import HuggingFacePipeline


class chatbot_chain():
    def __init__(self, model, tokenizer, user_id = None, db_connection = None, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        #self.user_id = user_id
        self.prompt = ChatPromptTemplate([

                                            ("system", "당신은 한국에 사는 외국인 노동자들을 위한 대출 상담 챗봇입니다. 사용자가 외국어로 물어봤을 때 해당 언어로 대답해야 합니다."),
                                            ("human", '{question}'),
                                            ("assistant", "{retriever}")
                                        ])

    
    def get_llmchain(self):
        tokenizer = self.tokenizer
        pad_token = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        eos_token = tokenizer.convert_tokens_to_ids("<eot_id>")
        gen_pipeline = pipeline(model = self.model, tokenizer = tokenizer, task = 'text-generation', return_full_text = False, max_new_tokens = 512, pad_token_id = pad_token, eos_token_id = eos_token)
        llm = HuggingFacePipeline(pipeline = gen_pipeline)
        prompt = self.prompt
        llm_chain = prompt | llm
        return llm_chain
    
    def get_chain_with_rag(self, dir_path, collection, k):
        vec_db = VectorStore.load_vectorstore(dir_path, collection)
        chain_retriever = retriever(vec_db, searched = k).get_retriever()
        llm_chain = self.get_llmchain()
        rag_context = {"question" : RunnablePassthrough(), "retriever" : chain_retriever}
        rag_chain = rag_context | llm_chain
        return rag_chain