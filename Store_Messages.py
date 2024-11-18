from langchain_community.chat_message_histories import SQLChatMessageHistory

class store_message():
    def __init__(self, session_id):
        self.session_id = session_id
    
    def store_history_sql(self):
        message_history = SQLChatMessageHistory(session_id = self.user_id, connection_string = 'sqlite///:chat_memory.db')
        