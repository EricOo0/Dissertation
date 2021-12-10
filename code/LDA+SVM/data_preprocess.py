import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
def data_process():
    label = pd.read_csv('../disaster_news_dataset.csv',usecols=['label'],encoding='utf-8',)
    content = pd.read_csv('../disaster_news_dataset.csv',usecols=['content'],encoding='utf-8')
    X_training, X_test, y_training, y_test = train_test_split(content,label , test_size=0.3,random_state=i,stratify=label)

    print("Test_Dataset example : ")
    print(X_test.columns)
    print(X_test['content'])
    print("Tset_Dataset       ---- len:{},shape:{}".format(X_test.shape[0],X_test.shape))
    print("Training_Dataset   ---- len:{},shape:{}".format(X_training.shape[0],X_training.shape))
    #print(type(y_test))
    training_set=pd.concat([y_training,X_training], axis=1)
    test_set=pd.concat([y_test,X_test], axis=1)
    training_set.to_csv("training_set.csv",index=False,encoding='utf-8')
    test_set.to_csv("test_set.csv",index=False,encoding='utf-8')
