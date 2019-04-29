from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pandas as pd

def score(gold_lab, test_lab):
    score = 0.0
    for (g,t) in zip(gold_lab, test_lab):
        if g == t:
            score+=0.25
            if g != 3:
                score+=0.5
        if g in [0,1,2] and t in [0,1,2]:
            score+=0.25
    
    return score

def fnc_score(actual, predicted):
    actual_score = score(actual, actual)
    calc_score = score(actual, predicted)
    return (calc_score*100)/actual_score

path = '/home/aman/Desktop/fnc/SMAI-fnc/fnc/input_data'

pickle_in = open(path+'/wv_test.pickle',"rb")
wv_test = pickle.load(pickle_in)   
pickle_in.close()
wv_test = np.array(wv_test)

pickle_in = open(path+'/wv_train.pickle',"rb")
wv_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(path+'/baseline_train.pickle',"rb")
base_train = pickle.load(pickle_in)   
pickle_in.close()

pickle_in = open(path+'/baseline_test.pickle',"rb")
base_test = pickle.load(pickle_in)   
pickle_in.close()

pickle_in = open(path+'/trainLab.pickle',"rb")
trainLabels = pickle.load(pickle_in)   
pickle_in.close()

pickle_in = open(path+'/testLab.pickle',"rb")
testLabels = pickle.load(pickle_in)   
pickle_in.close()

pickle_in = open(path+'/sentiment_train.pickle',"rb")
sentiment_train = pickle.load(pickle_in)   
pickle_in.close()

pickle_in = open(path+'/sentiment_test.pickle',"rb")
sentiment_test = pickle.load(pickle_in)   
pickle_in.close()

cat_trainLabels = []
for x in trainLabels:
    arr = [0] * 4
    arr[x[0]] = 1
    cat_trainLabels.append(arr)

trainLabels = trainLabels.reshape(trainLabels.shape[0])

sentiment_train, sentiment_val, wv_train, wv_val, base_train, base_val, trainLabels, valLabels = train_test_split(sentiment_train, wv_train, base_train, trainLabels, test_size=0.20, random_state=42)

input1 = Input(shape=(8,))
x1 = Dense(40, activation='relu')(input1)

input2 = Input(shape=(600,))
x2 = Dense(1000, activation='relu')(input2)
x2 = Dropout(0.25)(x2)
x2 = Dense(500, activation='relu')(x2)
x2 = Dropout(0.125)(x2)
x2 = Dense(500, activation='relu')(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(100, activation='relu')(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(40, activation='relu')(x2)

input3 = Input(shape=(18,))
x3 = Dense(40, activation='relu')(input3)

added = tf.keras.layers.add([x1, x2, x3])

out = Dense(4, activation='softmax')(added)
model = Model(inputs=[input1, input2, input3], outputs=out)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cat_trainLabels = to_categorical(trainLabels)
cat_valLabels = to_categorical(valLabels)
model.fit([sentiment_train, wv_train, base_train], cat_trainLabels, epochs=10, validation_data=([sentiment_val, wv_val, base_val], cat_valLabels))
valPred = model.predict([sentiment_val, wv_val, base_val])
testPred = model.predict([sentiment_test, wv_test, base_test])
argmaxValPred = np.argmax(valPred, axis=1)
argmaxTestPred = np.argmax(testPred, axis=1)
print('Accuracy over validation set: ', accuracy_score(argmaxValPred, valLabels))
print('Accuracy over test set: ', accuracy_score(argmaxTestPred, testLabels))
print('Fnc score over validation set: ', fnc_score(valLabels, argmaxValPred))
print('Fnc score over test set: ', fnc_score(testLabels, argmaxTestPred))

y_pred = pd.Series(argmaxValPred)
y_true = pd.Series(valLabels)
print('Validation Set confusion matrix is: ')
print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

testLabels = testLabels.reshape(testLabels.shape[0])
y_pred = pd.Series(argmaxTestPred)
y_true = pd.Series(testLabels)
print('Test Set confusion matrix is: ')
print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))