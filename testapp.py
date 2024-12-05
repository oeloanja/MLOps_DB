from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route('/borrow', methods = ['GET'])
def GetUserIDBorrow():
    user_id = request.args.get('userID')

    #user_token = request.args.get('user_token')
    # SECRET Key를 이용해 user 정보가 들어있는 json으로 파싱한 후 user 정보 얻기
    #user_id = user_token["userid"]
    
    return user_id

if __name__ == "__main__":
    app.run(debug=True)