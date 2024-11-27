from flask import request, jsonify



def GetUserIDBorrow():
    user_id = request.args.get('userID')

    #user_token = request.args.get('user_token')
    # SECRET Key를 이용해 user 정보가 들어있는 json으로 파싱한 후 user 정보 얻기
    #user_id = user_token["userid"]
    
    return user_id

def GetUserIDInvest():
    user_id = request.args.get('userID')
    return user_id
