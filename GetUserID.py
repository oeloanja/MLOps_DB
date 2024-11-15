from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get("user_id")
    message = request.json.get("message")
    print(f"User ID: {user_id}, Message: {message}")
    return jsonify({"reply": f"Hello, user {user_id}!"})

if __name__ == '__main__':
    app.run(port=5000)