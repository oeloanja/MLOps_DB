import flask
from flask import Flask, request
import loan_screening_pdf_loader as ll

app = Flask(__name__)

@app.route('/api/v1/screening/income', methods = ['POST'])
def get_screening_input():
    data = request.get_json()
    url = data['url']
    income = ll.income(url)
    print("Extracted income data:", income)
    result = {"income" : income}
    print("Response payload:", result)
    return result

@app.route('/api/v1/screening/length', methods = ['POST'])
def get_screening_length():
    data = request.get_json()
    url = data['url']
    length = ll.emp_length(url)
    print("Extracted income data:", length)
    result = {"emp_length" : length}
    print("Response payload:", result)
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 4000, debug = True)