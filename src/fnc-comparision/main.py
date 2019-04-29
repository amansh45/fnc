import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from sklearn import feature_extraction
import math
from features import word2vec_features, createWord2VecDict, sentiment_features
from features import word_overlap_features, refuting_features, polarity_features
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from score import report_score, LABELS, score_submission
from deepModel import fnc_score, load_split_vectors, vectorsExist, implementDeepModel
import pickle

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
    
def retrieveHeadBody(dataset_type, isSmall):
    l = os.getcwd().split('/')
    l.pop()
    l.pop()
    if isSmall and dataset_type == 'train':
        file_head = '/'.join(l) + "/input_data/smallData.csv"
    else:
        file_head = '/'.join(l) + "/input_data/"+dataset_type+"_stances.csv"    
    file_body = '/'.join(l) + "/input_data/"+dataset_type+"_bodies.csv"
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
    return pHead, pBody, stance_ids, body_ids, new_lab

def prepare_train_data(isSmall):
    pHead, pBody, stance_ids, body_ids, new_lab = retrieveHeadBody('train', isSmall)
    trainHead, valHead, trainLab, valLab, idTrain, idVal = train_test_split(pHead, new_lab, stance_ids, test_size=0.20, random_state=42)

    
    valBody = []
    for fid in idVal:
        valBody.append(pBody[body_ids.tolist().index(fid)])
        
    trainBody = []
    for fid in idTrain:
        trainBody.append(pBody[body_ids.tolist().index(fid)])
    
    #tpHead, tpBody, tstance_ids, tbody_ids, tnew_lab = retrieveHeadBody('competition_test',isSmall)
    #createWord2VecDict(pHead, pBody, tpHead, tpBody)
    return trainHead, trainBody, trainLab, valHead, valBody, valLab

def prepare_test_data(isSmall):
    pHead, pBody, stance_ids, body_ids, new_lab = retrieveHeadBody('competition_test',isSmall)
    testBody = []
    for fid in stance_ids:
        testBody.append(pBody[body_ids.tolist().index(fid)])
    
    return pHead, testBody, new_lab

def execGradientBoosting(isSmall):
    if vectorsExist() and not isSmall:
        trainSentiment_feats, valSentiment_feats, testSentiment_feats, train_wvFeats, val_wvFeats, test_wvFeats, trainBase_feats, valBase_feats, testBase_feats, trainLabels, valLabels, testLabels = load_split_vectors()
    else:
        #tHeadLine, tBody, tLabels, vHeadLine, vBody, vLabels = prepare_data_folds()
        
        trainHeadLine, trainBody, trainLabels, valHeadLine, valBody, valLabels = prepare_train_data(isSmall)
        trainLabels = np.reshape(trainLabels,(len(trainLabels),1))
        valLabels = np.reshape(valLabels,(len(valLabels),1))
        
        print('Data prepared and loaded')
        trainSentiment_feats = sentiment_features(trainHeadLine, trainBody)
        print('Train sentiment features generated....')
        trainBase_feats = generate_features(trainHeadLine, trainBody)
        print('Train baseline features generated....')
        trainHead_wvfeats, trainBody_wvfeats = word2vec_features(trainHeadLine, trainBody)
        train_wvFeats = []
        for x in range(len(trainHead_wvfeats)):
            train_wvFeats.append(np.concatenate((trainHead_wvfeats[x], trainBody_wvfeats[x])))
        train_wvFeats = np.array(train_wvFeats)
        print('Train word2vec features generated....')
        
        
        valBase_feats = generate_features(valHeadLine, valBody)
        print('Validation baseline features generated....')
        valSentiment_feats = sentiment_features(valHeadLine, valBody)
        print('Validation sentiment features generated....')
        valHead_wvfeats, valBody_wvfeats = word2vec_features(valHeadLine, valBody)
        val_wvFeats = []
        for x in range(len(valHead_wvfeats)):
            val_wvFeats.append(np.concatenate((valHead_wvfeats[x], valBody_wvfeats[x])))
        val_wvFeats = np.array(val_wvFeats)
        print('Validation word2vec features generated....')
        
        testHeadLine, testBody, testLabels = prepare_test_data(isSmall)
        testHead_wvfeats, testBody_wvfeats = word2vec_features(testHeadLine, testBody)
        print('Test word2vec features generated....')
        testSentiment_feats = sentiment_features(testHeadLine, testBody)
        print('Test sentiment features generated....')
        testBase_feats = generate_features(testHeadLine, testBody)
        print('Test baseline features generated....')
        test_wvFeats = []
        for x in range(len(testHead_wvfeats)):
            test_wvFeats.append(np.concatenate((testHead_wvfeats[x], testBody_wvfeats[x])))
            
    
    
    train_X = np.hstack((train_wvFeats, trainSentiment_feats))
    train_X = np.hstack((train_X, trainBase_feats))
    val_X = np.hstack((val_wvFeats, valSentiment_feats))
    val_X = np.hstack((val_X, valBase_feats))
    
    test_X = np.hstack((test_wvFeats, testSentiment_feats))
    test_X = np.hstack((test_X, testBase_feats))
    
    clf = GradientBoostingClassifier(n_estimators=5, random_state=14128, verbose=True)
    clf.fit(train_X, trainLabels)
        
    test_pred = clf.predict_proba(test_X)
    test_p = np.argmax(test_pred, axis=1)
    testPredictions = [round(value) for value in test_p]
    val_pred = clf.predict_proba(val_X)
    val_p = np.argmax(val_pred, axis=1)
    valPredictions = [round(value) for value in val_p]
    
    
    accuracy = accuracy_score(valLabels, valPredictions)
    print("Accuracy over validation dataset: %.2f%%" % (accuracy * 100.0))
    print('Score over validation dataset set: ',fnc_score(valLabels, valPredictions))
    
    y_pred = pd.Series(valPredictions)
    valLabels = np.array(valLabels)
    valLabels = valLabels.reshape(valLabels.shape[0])
    y_true = pd.Series(valLabels)
    print('Confusion matrix over validation dataset: ')
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    
    
    accuracy = accuracy_score(testLabels, testPredictions)
    print("Accuracy over test dataset: %.2f%%" % (accuracy * 100.0))
    print('Score over test dataset set: ',fnc_score(testLabels, testPredictions))
    
    y_pred = pd.Series(testPredictions)
    testLabels = np.array(testLabels)
    testLabels = testLabels.reshape(testLabels.shape[0])
    y_true = pd.Series(testLabels)
    print('Confusion matrix over test data: ')
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    
    pred_path = '/home/aman/Desktop/fnc/SMAI-fnc/fnc/output_data/treePred.pickle'
    pickle_out = open(pred_path,"wb")
    pickle.dump(test_pred, pickle_out)
    pickle_out.close()
    implementDeepModel(trainSentiment_feats, valSentiment_feats, testSentiment_feats, train_wvFeats, val_wvFeats, test_wvFeats, trainBase_feats, valBase_feats, testBase_feats, trainLabels, valLabels, testLabels)


execGradientBoosting(False)



print('--------------------------------------------')
print('--------------------------------------------')
print('Removing some of the unrelated samples....')
execGradientBoosting(True)


    

