import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from sklearn import feature_extraction
import math
from features import word_overlap_features, refuting_features, polarity_features
from sklearn.ensemble import GradientBoostingClassifier

def generate_features(headline, body):
    X_overlap = word_overlap_features(headline, body)
    print('word overlap fets generated...')
    X_refuting = refuting_features(headline, body)
    print('refuting fets generated...')
    X_polarity = polarity_features(headline, body)
    print('polarity fets generated...')
    X = np.c_[X_polarity, X_refuting, X_overlap]
    return X

def preprocess(stances, bodies):
    processed_heads, processed_bodies = [], []
    
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
    
    #trainHead, valHead, trainLab, valLab, idTrain, idVal = train_test_split(pHead, new_lab, stance_ids, test_size=0.20, random_state=42)

    
    valBody = []
    for fid in idVal:
        valBody.append(pBody[body_ids.tolist().index(fid)])
        
    trainBody = []
    for fid in idTrain:
        trainBody.append(pBody[body_ids.tolist().index(fid)])
        
    
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


trainHeadLine, trainBody, trainLabels, valHeadLine, valBody, valLabels = prepare_train_data()
print('Data prepared and loaded')


trainFeats = generate_features(trainHeadLine, trainBody)
valFeats = generate_features(valHeadLine, valBody)

print('Features calculated successfully...')
clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
clf.fit(trainFeats, trainLabels)
valPredictions = clf.predict(valFeats)
valScore = fnc_score(valLabels, valPredictions)
print('Validation Score is: ', valScore)


testHeadLine, testBody, testLabels = prepare_test_data()
testFeats = generate_features(testHeadLine, testBody)
testPredictions = clf.predict(testFeats)
testScore = fnc_score(testLabels, testPredictions)
print('Test Score is: ',testScore)