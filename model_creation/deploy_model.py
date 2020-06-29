import pandas as pd
from tensorflow import keras

def readData(path, filename):
    data = pd.read_csv(path + filename, sep=',')
    targets = data['targets']
    data = data.drop([data.columns[0], 'time', 'AUX', 'targets'], axis=1)
    return data, targets

data, targets = readData('../../recordings/', 'muse_recording.csv')
inputs = data[data.columns]

model = keras.models.load_model('../saved_model/pipelineModel')

outputs = model.predict(inputs)
print(outputs)