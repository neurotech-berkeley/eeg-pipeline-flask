from flask import Flask
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
CORS(app)
socketio = SocketIO(app=app, cors_allowed_origins='*')

@socketio.on('connect')
def dataSent():
    print("Connected!")

@socketio.on('test')
def receiveTest(msg):
    msg = json.loads(msg)
    msg = msg["data"]
    print("Received test message: " + msg)
    emit('test-reply', msg.upper())

@socketio.on('eeg-stream')
def receiveStream(reading):
    reading = json.loads(reading)
    print("Received Stream Reading: " + str(reading))
    if (max(reading['samples']) >= 95):
        emit('blink-state', 'eyes_open')
    else:
        emit('blink-state', 'eyes_closed')

if __name__ == "__main__":
    socketio.run(app, debug=True)

# Import the rest of the app
# import upload_model
# import deploy_model
# import stream_connect
