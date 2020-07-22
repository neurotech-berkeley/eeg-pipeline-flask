from flask import Flask
from flask_httpauth import HTTPBasicAuth
from flask_socketio import SocketIO, send
from flask_cors import CORS

app = Flask(__name__)
auth = HTTPBasicAuth()

CORS(app)
socketio = SocketIO(app)
@app.route('/')
# @auth.verify_password
# def verify_password(username, password):
#     if username == 'otherserver' and password == '123':
#         return True
#     else:
#         return False

@socketio.on('message')
def receiveMessage(msg):
    print("Message: " + msg)

def sendMessage(msg):
    send(msg, broadcast=True)

if __name__ == "__main__":
    socketio.run(app)

# Import the rest of the app
import upload_model
import deploy_model
import stream_connect

