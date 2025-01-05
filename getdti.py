'''
dti를 구하는 모듈
DB에 있는 데이터들 중에서 사용자 핸드폰번호를 이용해 불러옴
이를 통해 특정 개인과 일치하는 dti를 불러올 수 있게됨

'''

import pymysql

conn = pymysql.connect(host = 'localhost', port = 3306, user = 'root', password='1234', db = 'mydata')

def _get_data(user_pn): # dti를 구할 때 필요한 데이터를 불러오는 함수.
    mortgage_debt = 0
    mortgage_repayment = 0
    installment = 0
    mortgage_term = 0
    cursor = conn.cursor()
    cursor.execute(f'SELECT mortgage_debt, mortgage_repayment, installment, mortgage_term FROM mydata.user_log WHERE user_pn="{user_pn}"')
    row = cursor.fetchall()
    row0 = row[0]
    mortgage_debt = row0[0]
    mortgage_debt = float(mortgage_debt)
    mortgage_repayment = row0[1]
    mortgage_repayment = float(mortgage_repayment)
    installment = row0[2]
    mortgage_term = row0[3]
    conn.close()
    return mortgage_debt, mortgage_repayment, installment, mortgage_term


def calculate_dti(user_pn, income):
    mortgage_debt, mortgage_repayment, installment, mortgage_term = _get_data(user_pn)
    dti = ((((mortgage_debt/mortgage_term) + mortgage_repayment) + installment) / income) * 100 # 실제 dti를 구하는 부분.
    return dti