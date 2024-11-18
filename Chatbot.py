from Retriever import retriever
import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from Store_Messages import store_message


class chatbot_chain():
    def __init__(self, model, tokenizer, user_id, retriever, db_connection, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.user_id = user_id
        self.retriever = retriever
        self.db_connection = db_connection
        self.prompt = ChatPromptTemplate([

                                            ("system", "당신은 한국에 사는 외국인 노동자들을 위한 대출 상담 챗봇입니다. 사용자가 외국어로 물어봤을 때 해당 언어로 답해야 합니다."),
                                            ("human", "{user_input}"),
                                            ("Context", "{context}")
                                        ])

    def get_chain(self):
        chain = self.retriever | self.prompt | self.model | StreamingStdOutCallbackHandler()
        return chain
    
    def input_tokenize(self, user_input):
        tokenized = self.tokenizer.apply_chat_template(
            user_input,
            add_generation_prompt = True,
            return_tensors = 'pt'
        )
        return tokenized
    
    def get_eos_token(self):
        fin_token = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id>")]
        return fin_token
    
    def generate_answer(self, user_input):
        gen_answer_almost = self.model.generate(
            model_input = self.input_tokenize(user_input = user_input),
            max_new_tokens = 512,
            eos_token_id = self.get_eos_token(),
            do_sample = True,
            repetition_penalty = 1.00
        )
        gen_answer = self.tokenizer.batch_decode(gen_answer_almost)
        return gen_answer