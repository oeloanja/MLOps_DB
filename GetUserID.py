from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/GetUserID', methods = ['GET', 'POST'])
def GetUserID():
    if request.method == 'GET':
        id = request.args['id']
        name = request.args.get('name')
        return jsonify(name, id)
    elif request.method == 'POST':
        id = request.form['id']
        name = request.form['name']
        return jsonify(name, id)