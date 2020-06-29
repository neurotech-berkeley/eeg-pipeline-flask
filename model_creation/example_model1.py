
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import to_categorical
from keras import backend as K

port = pd.read_csv('inf_data.csv', sep=',')
#lb = LabelBinarizer()

port.platform = port.platform.fillna('X')
port.youtube_subs = port.youtube_subs.fillna(0)
port.youtube_engagement = port.youtube_engagement.fillna(0)
port.instagram_followers = port.instagram_followers.fillna(0)
port.instagram_engagement = port.instagram_engagement.fillna(0)
port.facebook_followers = port.facebook_followers.fillna(0)
port.facebook_engagement = port.facebook_engagement.fillna(0)
port.twitter_followers = port.twitter_followers.fillna(0)
port.twitter_engagement = port.twitter_engagement.fillna(0)
port.blog_traffic = port.blog_traffic.fillna(0)
port.follower_country1 = port.follower_country1.fillna('X')
port.follower_country2 = port.follower_country2.fillna('X')
port.follower_country3 = port.follower_country3.fillna('X')
port.follower_city1 = port.follower_city1.fillna('X')
port.follower_city2 = port.follower_city2.fillna('X')
port.follower_city3 = port.follower_city3.fillna('X')
port.inf_topic1 = port.inf_topic1.fillna('X')
port.inf_topic2 = port.inf_topic2.fillna('X')
port.inf_topic3 = port.inf_topic3.fillna('X')
port.client_topic1 = port.client_topic1.fillna('X')
port.client_topic2 = port.client_topic2.fillna('X')
port.client_topic3 = port.client_topic3.fillna('X')
port.follower_age_bracket1 = port.follower_age_bracket1.fillna(-1)
port.follower_age_bracket2 = port.follower_age_bracket2.fillna(-1)
port.gender = port.gender.fillna('X')
port.follower_percent_male = port.follower_percent_male.fillna(-1)
port.follower_percent_female = port.follower_percent_female.fillna(-1)
port.follower_percent_other = port.follower_percent_other.fillna(-1)
port.age = port.age.fillna(-1)

'''for name in port.columns:
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
'''
hash_encoder = ce.HashingEncoder(cols=['platform', 'follower_country1', 'follower_country2', 'follower_country3', 'follower_city1', 'follower_city2', 'follower_city3', 'inf_topic1', 'inf_topic2', 'inf_topic3', 'client_topic1', 'client_topic2', 'client_topic3', 'follower_age_bracket1', 'follower_age_bracket2', 'gender'])



targets = port['price']

del port['price']

inputs = port[port.columns]

inputs = hash_encoder.fit_transform(inputs)

inputs, targets = shuffle(inputs, targets, random_state=42)

print(inputs.iloc[[5]].to_string())

sess = tf.Session()
K.set_session(sess)

def make_model(nodes1=10, nodes2=10):
    model = Sequential()
    model.add(Dense(nodes1, activation='relu', input_shape=(21,), name="inputLayer"))
    model.add(Dense(nodes2, activation='relu'))
    model.add(Dense(1, name="outputLayer"))
    model.compile(optimizer = "adam", loss='mean_squared_error', metrics=['accuracy'])
    return model



model = make_model(40, 20)
model.fit(inputs, targets, 8, 200, validation_split = 0.2, verbose=0)

#Use TF to save the graph model instead of Keras save model to load it in Golang
builder = tf.saved_model.builder.SavedModelBuilder("../app/utilities/ssModel")
#Tag the model, required for Go
builder.add_meta_graph_and_variables(sess, ["ssTag"])
builder.save()
sess.close()

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

