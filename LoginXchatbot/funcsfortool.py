import VectorStore
from retriever import Retriever
from langchain.tools.retriever import create_retriever_tool




dirpath = './MLOps_chatbot'
collection = 'testdb'
vec_db = VectorStore.load_vectorstore(dir_path = dirpath, collection_name = collection)
ret = Retriever(dirpath, collection, searched = 2)
agent_retirever = ret.as_retriever()

dir2 = './fin_vocab'
collection2 = 'findb'
vec_db2 = VectorStore.load_vectorstore(dir_path = dir2, collection_name = collection2)
ret2 = Retriever(dir2, collection2, searched = 2)
agent_retirever2 = ret2.as_retriever()



def retrieve():
    retriever_tool = create_retriever_tool(
        agent_retirever,
        name = 'local_retriever',
        description = '''서비스 사용법, 대출 방법, 투자 방법에 관한 문서들을 검색할 수 있는 검색기 도구입니다. 
        서비스 사용법, 대출 방법, 투자 방법에 대한 질문이 들어오면 사용합니다.
        예시
        - 나 대출 받는 방법이 궁금해.
        - 투자를 하고싶은데 어떻게 해.
        - Billit 어떻게 써?
        - 이 서비스는 뭐하는 서비스야?
        이런 질문에는 이 local retriever를 써 관련된 문서를 찾은 후 그 문서를 기반으로 답하세요.'''
    )
    return retriever_tool

def fin_retriever():
    retriever_tool = create_retriever_tool(
        agent_retirever2,
        name = 'financial_vocabulary',
        description = '''금융지식에 관한 문서들을 검색할 수 있는 검색기 도구입니다.
        금융지식 관련 질문이 들어오면 사용합니다.
        예시
        - 자료열람요구권이 뭐야?
        - 개인신용평가대응권이 뭐야?
        - 인지세가 뭐야?
        - 개인신용평가대응권을 못쓰는 상황이 어떻게 돼?
        - 인지세는 누가, 얼마나 부담해?
    '''
    )
    return retriever_tool
