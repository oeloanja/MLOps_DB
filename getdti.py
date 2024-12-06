import pymysql

conn = pymysql.connect(host = 'localhost', user = 'root', password='1234', db = 'mydata')

def _get_data(email):
    cursor = conn.cursor()
    cursor.execute('SELECT ')