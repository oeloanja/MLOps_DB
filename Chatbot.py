from Retriever import retriever
import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM


class chatbot_chain():
    def __init__(self, model, tokenizer, user_id,*args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.user_id = user_id

    