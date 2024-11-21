from langchain_community.chat_message_histories import SQLChatMessageHistory
import getuserid

class MessageHistory():
    def __init__(self, connectstring, table):
        self.userid = self._get_user_id()
        self.connectstring = connectstring
        self.table = table
    
    def get_func(func1, func2):
        def wrapper(*args, **kwargs):
            result = func1(*args, **kwargs)
            if result is None:
                result = func2(*args, **kwargs)
            return result
        return wrapper
    
    @get_func(getuserid.GetUserIDBorrow, getuserid.GetUserIDInvest)
    def _get_user_id():
        pass

    def store_message(self, messages):
        store_obj = SQLChatMessageHistory(session_id=self.userid, connection_string=self.connectstring, table_name=self.table)
        return store_obj.add_messages(messages)
    
    def get_messages(self):
        getter_obj = SQLChatMessageHistory(session_id=self.userid, connection_string=self.connectstring, table_name=self.table)
        getted = getter_obj.get_messages()
        return getted
