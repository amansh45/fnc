def word2vec_features(headline, body):
    l = os.getcwd().split('/')
    l.pop()
    l.pop()    
    try:
        model_path = '/'.join(l) + "/output_data/word2vec.pickle"
        print(model_path)
        pickle_in = open(model_path,"rb")
        word2vec_model = pickle.load(pickle_in)    
    except Exception as e:
        print(e)
        print('Dictionary for word2vec model does not exist')
        
    
    counter = 0
    for i, head in enumerate(headline):
        counter += len(head.split(' '))
    avg_headline_len = math.ceil(counter/i)
    
    avg_body_len = 30

    
    wordVec_head = []    
    for head in headline:
        split_head = head.split(' ')
        hVec = np.array([])
        if len(split_head) >= avg_headline_len:
            for x in range(avg_headline_len):
                word = split_head[x]
                hVec = np.concatenate((hVec, word2vec_model[word]))
        else:
            for x in range(len(split_head)):
                word = split_head[x]
                hVec = np.concatenate((hVec, word2vec_model[word]))
            for x in range(avg_headline_len - len(split_head)):
                zero_arr = np.zeros(300)
                hVec = np.concatenate((hVec, zero_arr))
        wordVec_head.append(hVec)
    
    wordVec_body = []    
    for b in body:
        split_body = b.split(' ')
        bVec = np.array([])
        if len(split_body) >= avg_body_len:
            for x in range(avg_body_len):
                word = split_body[x]
                bVec = np.concatenate((bVec, word2vec_model[word]))
        else:
            for x in range(len(split_body)):
                word = split_body[x]
                bVec = np.concatenate((bVec, word2vec_model[word]))
            for x in range(avg_body_len - len(split_body)):
                zero_arr = np.zeros(300)
                bVec = np.concatenate((bVec, zero_arr))
        wordVec_body.append(bVec)
        
    return wordVec_head, wordVec_body