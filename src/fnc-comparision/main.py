import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from sklearn import feature_extraction
import math
from features import word2vec_features, createWord2VecDict, sentiment_features
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras import layers
from score import report_score, LABELS, score_submission


def preprocess(stances, bodies):
    processed_heads, processed_bodies = [], []
    
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric, stop words.
    for headline in stances:
        clean_head = " ".join(re.findall(r'\w+', headline, flags=re.UNICODE)).lower()
        tok_head = [t for t in nltk.word_tokenize(clean_head)]
        pro_head = [w for w in tok_head if w not in feature_extraction.text.ENGLISH_STOP_WORDS and len(w) > 1 and (len(w)!=2 or w[0]!=w[1]) ]
        processed_heads.append(' '.join(pro_head))
        
    for body in bodies:
        clean_body = " ".join(re.findall(r'\w+', body, flags=re.UNICODE)).lower()
        tok_body = [t for t in nltk.word_tokenize(clean_body)]
        pro_body = [w for w in tok_body if w not in feature_extraction.text.ENGLISH_STOP_WORDS and len(w) > 1 and (len(w)!=2 or w[0]!=w[1]) ]
        processed_bodies.append(' '.join(pro_body))
    
    return processed_heads, processed_bodies

def cross_validation_split(dataset, folds=3):
    dataset_split = []
    dataset_copy = dataset.tolist()
    fold_size = math.ceil(int(len(dataset) / folds))
    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            fold.append(dataset_copy.pop())
        dataset_split.append(fold)
    return dataset_split
    

def prepare_train_data():
    l = os.getcwd().split('/')
    l.pop()
    l.pop()
    file_head = '/'.join(l) + "/input_data/train_stances.csv"
    file_body = '/'.join(l) + "/input_data/train_bodies.csv"
    head = pd.read_csv(file_head)
    body = pd.read_csv(file_body)
    head_array = head.values
    body_array = body.values
    labels = head_array[:,2]
    stance_ids = head_array[:,1]
    body_ids = body_array[:,0]
    new_lab = []
    for i in labels:
        if i == 'unrelated':
            new_lab.append(3)
        if i == 'agree':
            new_lab.append(0)
        if i == 'discuss':
            new_lab.append(2)
        if i == 'disagree':
            new_lab.append(1)
    
    pHead, pBody = preprocess(head_array[:,0], body_array[:,1])
    
    trainHead, valHead, trainLab, valLab, idTrain, idVal = train_test_split(pHead, new_lab, stance_ids, test_size=0.20, random_state=42)

    
    valBody = []
    for fid in idVal:
        valBody.append(pBody[body_ids.tolist().index(fid)])
        
    trainBody = []
    for fid in idTrain:
        trainBody.append(pBody[body_ids.tolist().index(fid)])
        
    #createWord2VecDict(pHead, pBody)
    
    return trainHead, trainBody, trainLab, valHead, valBody, valLab

def prepare_test_data():
    l = os.getcwd().split('/')
    l.pop()
    l.pop()
    file_head = '/'.join(l) + "/input_data/competition_test_stances.csv"
    file_body = '/'.join(l) + "/input_data/competition_test_bodies.csv"
    head = pd.read_csv(file_head)
    body = pd.read_csv(file_body)
    head_array = head.values
    body_array = body.values
    labels = head_array[:,2]
    stance_ids = head_array[:,1]
    body_ids = body_array[:,0]
    new_lab = []
    for i in labels:
        if i == 'unrelated':
            new_lab.append(3)
        if i == 'agree':
            new_lab.append(0)
        if i == 'discuss':
            new_lab.append(2)
        if i == 'disagree':
            new_lab.append(1)
    
    pHead, pBody = preprocess(head_array[:,0], body_array[:,1])
    
    testBody = []
    for fid in stance_ids:
        testBody.append(pBody[body_ids.tolist().index(fid)])
        
    
    return pHead, testBody, new_lab

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

#tHeadLine, tBody, tLabels, vHeadLine, vBody, vLabels = prepare_data_folds()

trainHeadLine, trainBody, trainLabels, valHeadLine, valBody, valLabels = prepare_train_data()
trainLabels = np.reshape(trainLabels,(len(trainLabels),1))
valLabels = np.reshape(valLabels,(len(valLabels),1))

print('Data prepared and loaded')

#trainHead_wvfeats, trainBody_wvfeats = word2vec_features(trainHeadLine, trainBody)
trainSentiment_feats = sentiment_features(trainHeadLine, trainBody)
print('Train word2vec features generated....')

#valHead_wvfeats, valBody_wvfeats = word2vec_features(valHeadLine, valBody)
valSentiment_feats = sentiment_features(valHeadLine, valBody)
print('Validation word2vec features generated....')

# =============================================================================
# train_wvFeats = []
# for x in range(len(trainHead_wvfeats)):
#     train_wvFeats.append(np.concatenate((trainHead_wvfeats[x], trainBody_wvfeats[x])))
# train_wvFeats = np.array(train_wvFeats)
# 
# val_wvFeats = []
# for x in range(len(valHead_wvfeats)):
#     val_wvFeats.append(np.concatenate((valHead_wvfeats[x], valBody_wvfeats[x])))
# val_wvFeats = np.array(val_wvFeats)
# 
# train_X = np.hstack((train_wvFeats, trainSentiment_feats))
# val_X = np.hstack((val_wvFeats, valSentiment_feats))
# =============================================================================


clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
clf.fit(trainSentiment_feats, trainLabels)



from sklearn.metrics import accuracy_score

#model.fit(trainSentiment_feats, trainLabels)
#model.fit(trainSentiment_feats, trainLabels)
# make predictions for test data

val_pred = clf.predict(valSentiment_feats)
valPredictions = [round(value) for value in val_pred]
# evaluate predictions
accuracy = accuracy_score(valLabels, valPredictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('Score over validation set: ',fnc_score(valLabels, valPredictions))













# =============================================================================
# print('Shape of train feats',np.array(train_wvFeats).shape)
# print('Shape of validation feats',np.array(val_wvFeats).shape)
# print('Features calculated successfully...')
# 
# 
# model = tf.keras.Sequential()
# model.add(layers.Dense(1000, activation='relu'))
# model.add(layers.Dense(500, activation='relu'))
# model.add(layers.Dense(100, activation='relu'))
# model.add(layers.Dense(4, activation='softmax'))
# 
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# 
# model.fit(train_wvFeats, trainLabels, epochs=10, batch_size=32,validation_data=(val_wvFeats, valLabels))
# 
# 
# # =============================================================================
# # clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
# # clf.fit(train_wvFeats, trainLabels)
# # =============================================================================
# valPredictions = model.predict(val_wvFeats)
# 
# valScore = fnc_score(valLabels, valPredictions)
# print('Validation Score is: ', valScore)
# =============================================================================




testHeadLine, testBody, testLabels = prepare_test_data()
#testHead_wvfeats, testBody_wvfeats = word2vec_features(testHeadLine, testBody)
print('Test word2vec features generated....')


testSentiment_feats = sentiment_features(testHeadLine, testBody)

# =============================================================================
# test_wvFeats = []
# for x in range(len(testHead_wvfeats)):
#     test_wvFeats.append(np.concatenate((testHead_wvfeats[x], testBody_wvfeats[x])))
#     
# test_X = np.hstack((test_wvFeats, testSentiment_feats))
# 
# 
# =============================================================================

test_pred = clf.predict(testSentiment_feats)
testPredictions = [round(value) for value in test_pred]
# evaluate predictions
accuracy = accuracy_score(testLabels, testPredictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('Score over test set: ',fnc_score(testLabels, testPredictions))




y_pred = pd.Series(testPredictions)
y_true = pd.Series(testLabels)
print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

