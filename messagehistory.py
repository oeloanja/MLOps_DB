from langchain_community.chat_message_histories import SQLChatMessageHistory
import getuserid

class MessageHistory():
    def __init__(self, connectstring, table):
        self.userid = getuserid.GetUserIDBorrow()
        if self.userid == None:
            self.userid = getuserid.GetUserIDInvest()
        self.connectstring = connectstring
        self.table = table
    
    def store_message(self, messages):
        store_obj = SQLChatMessageHistory(session_id=self.userid, connection_string=self.connectstring, table_name=self.table)
        return store_obj.add_messages(messages)
    
    def get_messages(self):
        getter_obj = SQLChatMessageHistory(session_id=self.userid, connection_string=self.connectstring, table_name=self.table)
        getted = getter_obj.get_messages()
    
