import pickle as pk
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from deepModel import fnc_score
import os

def finalScore():
    path = os.getcwd().split('/')
    path.pop()
    path.pop()
    
    
    f=open(path+'/output_data/deepPred.pickle','rb')
    deepPred= pk.load(f)
    f.close()
    
    f=open(path+'/output_data/treePred.pickle','rb')
    treePred= pk.load(f)
    f.close()
    
    
    f=open(path+'/output_data/testLabs.pickle','rb')
    testLabs= pk.load(f)
    f.close()
    
    
    res= deepPred*0.5 + treePred*0.5
    
    
    deep_pred=[]
    for i in deepPred:
    	deep_pred.append(np.argmax(i))
    
    
    tree_pred=[]
    for i in treePred:
    	tree_pred.append(np.argmax(i))
    
    y_pred=[]
    for i in res:
    	y_pred.append(np.argmax(i))
    
    
    y_pred=np.array(y_pred)
    y_actual=testLabs
    deep_pred=np.array(deep_pred)
    tree_pred=np.array(tree_pred)
    
    
    print(accuracy_score(y_actual, y_pred)*100)
    
    
    y_pred = pd.Series(y_pred)
    y_true = pd.Series(y_actual)
    print('Confusion matrix over test data: ')
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    
    print("score= ",fnc_score(y_actual,y_pred))
    print("precision=",precision_score(y_actual,y_pred,average='weighted'))
    print("recall= ",recall_score(y_actual,y_pred,average='weighted'))
    print("f1 score= ",f1_score(y_actual,y_pred,average='weighted'))