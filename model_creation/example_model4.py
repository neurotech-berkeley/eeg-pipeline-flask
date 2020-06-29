
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import to_categorical

port = pd.read_csv('pricing.csv', sep=',')
lb = LabelBinarizer()

#port.Story = port.Story.fillna(0)
port.Segment2 = port.Segment2.fillna('X')
port.Age = port.Age.fillna(0)
port.Gender = port.Gender.fillna('X')
port.Client = port.Client.fillna('X')

del port['Story']
#del port['Segment']
#del port['Segment2']
#del port['Platform']
#del port['Age']
#del port['Gender']
#del port['Client']

for name in port.columns:
    if not isinstance(port[name][0], np.int64) and not isinstance(port[name][0], np.float64):
        temp = lb.fit_transform(port[name])
        cols = []
        if len(lb.classes_) > 2:
            for colname in lb.classes_:
                cols.append(name + ':' + str(colname))
        elif len(lb.classes_) == 1:
            cols.append(name + ':' + lb.classes_[0])
        else:
            cols.append(name + ':' + lb.classes_[1] + ' 0 = ' + lb.classes_[0])
        temp2 = pd.DataFrame(temp, columns= cols)
        port = pd.concat([port, temp2], axis= 1)
        del port[name]

testInput = port.iloc[[port['Followers'].size - 1]]
port = port.drop([port['Followers'].size - 1], axis=0)

#port = port[port['G3'] != 1]

targets = port['Post']

del port['Post']
del testInput['Post']

inputs = port[port.columns]

inputs, targets = shuffle(inputs, targets, random_state=42)

#targets_ohl = keras.utils.to_categorical(targets, num_classes=4)

def make_multi_model(nodes1=10, nodes2=10):
    model = Sequential()
    model.add(Dense(nodes1, activation='relu', input_shape=(42,)))
    model.add(Dense(nodes2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer = "adam", loss='mean_squared_error', metrics=['accuracy'])
    return model


#keras_clf = KerasClassifier(make_multi_model, nodes1=15, nodes2=15, epochs=200, batch_size=6, verbose=0)
estimator = KerasRegressor(build_fn=make_multi_model, epochs=200, batch_size=8, nodes1=40, nodes2=20, verbose=0)

kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(estimator, inputs, targets, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
estimator.fit(inputs, targets)

print(estimator.predict(testInput))

# print(cross_val_score(keras_clf, inputs, targets, cv=3, scoring='accuracy'))

# score = keras_clf.evaluate(inputs, targets_ohl)
# outputs = keras_clf.predict(inputs)
# print('Loss:', score[0])
# print('Accuracy:', score[1])

#grid = GridSearchCV(estimator, n_jobs=-1, cv=3, param_grid={'epochs':[200],'batch_size':[8],'nodes1':[40],'nodes2':[20]})

#grid.fit(inputs, targets)
#print('The parameters of the best model are: ') 
# print(grid.best_params_)

#grid_results = grid.cv_results_ 
#zipped_results = list(zip(grid_results["params"], grid_results["mean_test_score"])) 
#zipped_results_sorted = sorted(zipped_results, key=lambda tmp : tmp[1])[::-1]
#for element in zipped_results_sorted: 
#    print(element[1], element[0])

