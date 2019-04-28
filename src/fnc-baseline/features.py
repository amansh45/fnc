import nltk
import numpy as np


def word_overlap_features(headlines, bodies):
    X = []
    for (head, body) in zip(headlines, bodies):
        tokenize_headline = [t for t in nltk.word_tokenize(head)]
        tokenize_body = [t for t in nltk.word_tokenize(body)]
        features = [
            len(set(tokenize_headline).intersection(tokenize_body)) / float(len(set(tokenize_headline).union(tokenize_body)))]
        X.append(features)
    return np.array(X)

def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for (head, body) in zip(headlines, bodies):
        tokenize_headline = [t for t in nltk.word_tokenize(head)]
        #tokenize_body = [t for t in nltk.word_tokenize(body)]
        head_features = [1 if word in tokenize_headline else 0 for word in _refuting_words]
        #body_features = [1 if word in tokenize_body else 0 for word in _refuting_words]
        X.append(head_features)
        #X.append(body_features)
    return np.array(X)


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = [t for t in nltk.word_tokenize(text)]
        return sum([t in _refuting_words for t in tokens]) % 2
    
    X = []
    for (headline, body) in zip(headlines, bodies):
        features = []
        features.append(calculate_polarity(headline))
        features.append(calculate_polarity(body))
        X.append(features)
    return np.array(X)

