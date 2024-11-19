from langchain_community.chat_message_histories import SQLChatMessageHistory

class store_message():
    def __init__(self, session_id):
        self.session_id = session_id
    
    def get_store_sql(self):
        return SQLChatMessageHistory(session_id = self.session_id, connection_string = 'sqlite///:chat_memory.db')

    def store_history_sql(self, messages):
        return self.get_store_sql().add_messages(messages = messages)
    
    def retrive_messages(self):
        retrived = self.get_store_sql().aget_messages()
        return retrived
    
    def get_messages(self):
        getted = self.get_store_sql().get_messages()
        return getted