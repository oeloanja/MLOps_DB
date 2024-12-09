import pymysql

conn = pymysql.connect(host = 'localhost', user = 'root', password='1234', db = 'mydata')

def _get_data(email):
    cursor = conn.cursor()
    cursor.execute('SELECT mortgage_debt, mortgage_repayment, installment, mortgage_term FROM user_log WHERE user_id={email}')
    while True:
        row = cursor.fetchone()
        if row == None:
            break
        mortgage_debt = row[0]
        mortgage_repayment = row[1]
        installment = row[2]
        mortgage_term = row[3]
    conn.close()
    return mortgage_debt, mortgage_repayment, installment, mortgage_term

def calculate_dti(email, income):
    mortgage_debt, mortgage_repayment, installment, mortgage_term = _get_data(email)
    dti = ((((mortgage_debt//mortgage_term) + mortgage_repayment) + installment) / income) * 100
    return dti