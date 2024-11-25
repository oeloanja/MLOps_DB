from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['USER_DATABASE_URI'] = 'dburiborrow'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


@app.route('/borrow', methods = ['GET'])
def GetUserIDBorrow():
    user_id = request.args.get('userID')

    #user_token = request.args.get('user_token')
    # SECRET Key를 이용해 user 정보가 들어있는 json으로 파싱한 후 user 정보 얻기
    #user_id = user_token["userid"]
    
    return user_id

@app.route('/invest', methods = ['GET'])
def GetUserIDInvest():
    user_id = request.args.get('userID')
    return user_id

class UserBorrow(db.Model):
    __tablename__ = 'user_borrow'
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String, nullable = False)

class UserInvest(db.Model):
    __tablename__ = 'user_invest'
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String, nullable = False)

@app.route('/api/users/borrow/mypage', methods = ['GET'])
def GetUserNameBorrow(userid = None):
    if userid is None:
        userid = GetUserIDBorrow()
    user = UserBorrow.query.get(userid)
    if user:
        user_name = user.username
        return user_name
    else:
        return None

@app.route('/api/users/invest/mypage', methods = ['GET'])
def GetUserNameInvest(userid = None):
    if userid is None:
        userid = GetUserIDInvest()
    user = UserInvest.query.get(userid)
    if user:
        user_name = user.username
        return user_name
    else:
        return None
