import nltk
import numpy as np
import os
import gensim
from nltk.tokenize import regexp_tokenize
import itertools
import math
from gensim.models import Word2Vec
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def createWord2VecDict(headline, body):
    l = os.getcwd().split('/')
    l.pop()
    l.pop()
    model_path = '/'.join(l) + "/input_data/GoogleNews-vectors-negative300.bin"
    
    print('Loading Pretrained word2vec model...')
    wordVec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    print('Word2vec model trained successfully...')
    sent1 = []
    sent1.extend(headline)
    sent1.extend(body)
    START = '$_START_$'
    END = '$_END_$'
    sentence = ["%s %s %s" % (START,x,END) for x in sent1]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
                    
    freq = nltk.FreqDist(itertools.chain(*tokenize_sent))
    print('found ',len(freq),' unique words')
    vocab = freq.most_common(30000 - 1)
    index_to_word = [x[0] for x in vocab]
    
    self_word2vec_model = {}
    
    for i,j in enumerate(index_to_word):
        try:
            self_word2vec_model[j] = wordVec_model[j]
        except KeyError:
            self_word2vec_model[j] = np.zeros(300)
    
    model_path = '/'.join(l) + "/output_data/word2vec.pickle"
    pickle_out = open(model_path,"wb")
    pickle.dump(self_word2vec_model, pickle_out)
    pickle_out.close()
    
def retrieveTFIDFScore(data):
    cv = TfidfVectorizer()
    X = cv.fit_transform(data)
    return cv.vocabulary_
    
def word2vec_features(headline, body):
    l = os.getcwd().split('/')
    l.pop()
    l.pop()    
    try:
        model_path = '/'.join(l) + "/output_data/word2vec.pickle"
        print(model_path)
        pickle_in = open(model_path,"rb")
        word2vec_model = pickle.load(pickle_in)   
        pickle_in.close()
    except Exception as e:
        print(e)
        print('Dictionary for word2vec model does not exist')
        

    wordVec_head = []    
    for head in headline:
        split_head = head.split(' ')
        hVec = np.zeros(300)
        
        count = 0
        for x in range(len(split_head)):
            word = split_head[x]
            try:
                hVec = np.add(hVec, word2vec_model[word])
            except KeyError:
                pass
            count += 1
            
        hVec = np.multiply(hVec, (1/count))
        
        wordVec_head.append(hVec)
    
    
    wordVec_body = []    
    for b in body:
        split_body = b.split(' ')
        bVec = np.zeros(300)
        
        count = 0
        for x in range(len(split_body)):
            word = split_body[x]
            try:
                bVec = np.add(bVec, word2vec_model[word])
            except KeyError:
                pass
            count += 1
            
        hVec = np.multiply(hVec, (1/count))
        
        wordVec_body.append(bVec)

    return wordVec_head, wordVec_body


def sentiment_features(headline, body):
    sid = SentimentIntensityAnalyzer()
    sentiment_feats = []
    for x in range(len(headline)):
        single_head = headline[x]
        single_body = body[x]
        head_sentiment = sid.polarity_scores(single_head)
        body_sentiment = sid.polarity_scores(single_body)
        sentiment_feats.append([head_sentiment['neg'],head_sentiment['neu'],head_sentiment['pos'],head_sentiment['compound'],
                               body_sentiment['neg'],body_sentiment['neu'],body_sentiment['pos'],body_sentiment['compound']])
    
    return np.array(sentiment_feats)
    