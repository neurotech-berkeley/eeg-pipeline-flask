from flask import Flask
from flask_socketio import SocketIO, send, emit
from engineio.payload import Payload
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from tensorflow import keras
import pandas as pd
import json
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
auth = HTTPBasicAuth()
app.config['SECRET_KEY'] = 'mysecret'
CORS(app)
Payload.max_decode_packets = 1000000
socketio = SocketIO(app=app, cors_allowed_origins='*')
model = None

@socketio.on('connect')
def dataSent():
    print("Connected!")
    print("Loading Model")
    global model
    model = keras.models.load_model('./saved_models/test_user/pipelineModel_test')
    print("Loaded")

@socketio.on('test')
def receiveTest(msg):
    msg = json.loads(msg)
    msg = msg["data"]
    print("Received test message: " + msg)
    emit('test-reply', msg.upper())

# @socketio.on('eeg-stream')
# def receiveStream(reading):
#     print(reading)
#     emit('response', 'Ack')
    # reading = json.loads(reading)
    # print("Received Stream Reading: " + str(reading))
    # if (max(reading['samples']) >= 95):
    #     emit('blink-state', 'eyes_open')
    # else:
    #     emit('blink-state', 'eyes_closed')

@socketio.on('eeg-stream')
def model_streamed(reading):
    reading = EEGReading(reading)
    inputs = pd.DataFrame([reading[2:6]], dtype=float)
    print(inputs)

    outputs = model.predict(inputs)
    print(outputs)

@auth.verify_password
def verify_password(username, password):
    if username == 'otherserver' and password == '123':
        return True
    else:
        return False

def processBlink(reading):
    if (max(reading.samples) >= 300):
        emit('blink-state', 'eyes_open')

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
