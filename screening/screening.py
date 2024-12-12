import flask
from flask import Flask, request
import loan_screening_pdf_loader as ll

app = Flask(__name__)

@app.route('/api/v1/screening/income', methods = ['POST'])
def get_screening_input():
    data = request.get_json()
    url = data['url']
    income = ll.income(url)
    result = {"income" : income}
    return result

@app.route('/api/v1/screening/length', methods = ['POST'])
def get_screening_length():
    data = request.get_json()
    url = data['url']
    length = ll.emp_length(url)
    result = {"emp_length" : length}
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 4000, debug = True)