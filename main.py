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
    reading = EEGReading(reading)
    print("Received Stream Reading: " + str(reading))
    processJawClench(reading)  

def processBlink(reading):
    if (max(reading.samples) >= 300):
        emit('blink-state', 'eyes_open')
    else:
        emit('blink-state', 'eyes_closed')

def processJawClench(reading):
    if (max(reading.samples) >= 250):
        emit('jaw-state', 'clenched')
    # else:
    #     emit('jaw-state', 'relaxed')

if __name__ == "__main__":
    socketio.run(app, debug=True)

# Import the rest of the app
# import upload_model
# import deploy_model
# import stream_connect

class EEGReading:
    def __init__(self, reading):
        reading = json.loads(reading)
        self.electrode = reading['electrode']
        self.timestamp = reading['timestamp']
        self.samples = reading['samples']
