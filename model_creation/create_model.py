import numpy as np
import pandas as pd
import time
# import category_encoders as ce
# from sklearn import datasets
# from sklearn.utils import shuffle
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
import tensorflow as tf
# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import RMSprop
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.utils import to_categorical
from keras import backend as K

def readData(path, filename):
    data = pd.read_csv(path + filename, sep=',')
    targets = data['targets']
    data = data.drop([data.columns[0], 'time', 'AUX', 'targets'], axis=1)
    return data, targets

data, targets = readData('../../recordings/', 'muse_recording.csv')
inputs = data[data.columns]

# Do we shuffle inputs if they are linear in time? ML help pls
# inputs, targets = shuffle(inputs, targets, random_state=42)

def make_model(shape, nodes1=10, nodes2=10):
    model = Sequential()
    model.add(Dense(nodes1, activation='relu', input_shape=(shape,), name="inputLayer"))
    model.add(Dense(nodes2, activation='relu'))
    model.add(Dense(1, name="outputLayer"))
    model.compile(optimizer = "adam", loss='mean_squared_error', metrics=['accuracy'])
    return model

model = make_model(data.shape[1])
model.fit(inputs, targets, batch_size = 20, epochs = 20, validation_split = 0.2, verbose=1)


model.save("../saved_model/pipelineModel")

#print(cross_val_score(model, inputs, targets, cv=3, scoring='accuracy'))

# score = model.evaluate(inputs, targets)
# outputs = model.predict(inputs)
# print('Loss:', score[0])
# print('Accuracy:', score[1])
# print(outputs)

# grid = GridSearchCV(estimator, n_jobs=-1, cv=3, param_grid={'epochs':[200],'batch_size':[8],'nodes1':[40],'nodes2':[20]})

# grid.fit(inputs, targets)
# print('The parameters of the best model are: ') 
# print(grid.best_params_)

# grid_results = grid.cv_results_ 
# zipped_results = list(zip(grid_results["params"], grid_results["mean_test_score"])) 
# zipped_results_sorted = sorted(zipped_results, key=lambda tmp : tmp[1])[::-1]
# for element in zipped_results_sorted: 
#    print(element[1], element[0])




