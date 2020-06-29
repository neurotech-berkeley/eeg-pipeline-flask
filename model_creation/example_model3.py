import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier

titanic = pd.read_csv('data/train.csv')

del titanic['Name']

titanic.Age = titanic.Age.fillna(28.00)
titanic.Cabin = titanic.Cabin.fillna('X')
titanic.Embarked = titanic.Embarked.fillna('X')

titanic['Cabin'] = titanic['Cabin'].str[0]

lb = LabelBinarizer()

for name in titanic.columns:
    if not isinstance(titanic[name][0], np.int64) and not isinstance(titanic[name][0], np.float64):
        temp = lb.fit_transform(titanic[name])
        cols = []
        if len(lb.classes_) > 2:
            for colname in lb.classes_:
                cols.append(name + ':' + str(colname))
        else:
            cols.append(name + ':' + lb.classes_[1] + ' 0 = ' + lb.classes_[0])
        temp2 = pd.DataFrame(temp, columns= cols)
        port = pd.concat([titanic, temp2], axis= 1)
        del titanic[name]

targets = titanic['Survived']
del titanic['Survived']

inputs = titanic[titanic.columns]

inputs, targets = shuffle(inputs, targets, random_state=42)
#kernel="poly", degree=2, coef0=1, C=5
# svc = SVC()
# print(svc.score(inputs, targets))

tree = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 6, min_samples_split = 19, min_weight_fraction_leaf = 0)

# def make_svc_model(kernel='poly', ):
#     model = Sequential()
#     model.add(Dense(1, activation='sigmoid', input_shape=(6,)))
#     sgd = optimizers.SGD(lr=learn, momentum=0.0, decay=0.0, nesterov=False)
#     model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# keras_sgd = KerasClassifier(make_sgd_model, epochs=400, batch_size=10, verbose=0)

def make_sgd_model(learn=.001):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(6,)))
    sgd = optimizers.SGD(lr=learn, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

keras_sgd = KerasClassifier(make_sgd_model, epochs=400, batch_size=10, verbose=0)

def make_multi_model(nodes1=50, nodes2=50):
    model = Sequential()
    model.add(Dense(nodes1, activation='relu', input_shape=(6,)))
    model.add(Dense(nodes2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

keras_clf = KerasClassifier(make_multi_model, epochs=400, batch_size=7, nodes1=350, nodes2=150, verbose=0)

democracy = VotingClassifier(estimators=[('neural',keras_clf),('tree',tree)], voting='soft')

print("Comparison CV Accuracy:\n")
for clf in (keras_clf, tree, keras_sgd, democracy):
    scores = cross_val_score(clf, inputs, targets, cv=4, scoring="accuracy")
    print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), clf.__class__.__name__))

# grid = GridSearchCV(keras_sgd, n_jobs=-1, cv=4, param_grid={'epochs':[400],'batch_size':[12],'learn':[.001]})

# grid.fit(inputs, targets)
# print('The parameters of the best model are: ') 

# grid_results = grid.cv_results_ 
# zipped_results = list(zip(grid_results["params"], grid_results["mean_test_score"])) 
# zipped_results_sorted = sorted(zipped_results, key=lambda tmp : tmp[1])[::-1]
# for element in zipped_results_sorted: 
#     print(element[1], element[0])
