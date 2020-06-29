from main import app, auth
from flask import request
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
from tensorflow import keras

# TODO: Change function so it takes a stream of data, runs it through the loaded model and returns a stream of data
# Likely this will be split into multiple functions, one to load the model and the other to stream to it but not sure right now
@app.route('/deploy_model', methods=['POST']) # As above unclear whether this should be post, if changed, request.form (below) will need to be changed
@auth.login_required # Username/password in main.py for now
def deploy_model():
    def readData(path, filename):
        data = pd.read_csv(path + filename, sep=',')
        targets = data['targets']
        data = data.drop([data.columns[0], 'time', 'AUX', 'targets'], axis=1)
        return data, targets

    data, targets = readData('../recordings/', 'muse_recording.csv')
    inputs = data[data.columns]

    # Example of loading a model from a filepath, currently from ./saved_models/<username>/<model name>, this is how they are saved
    print('./saved_models/' + secure_filename(request.form['user']) + '/' + secure_filename(request.form['model_name']))
    model = keras.models.load_model('./saved_models/' + secure_filename(request.form['user']) + '/' + secure_filename(request.form['model_name']))

    outputs = model.predict(inputs)
    return str(outputs)