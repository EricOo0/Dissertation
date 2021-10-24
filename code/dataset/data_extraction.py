import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
def a(i):
    origin= pd.read_csv('/data2/wzf/dataset/disaster_new_20200828.csv',usecols=['label','content'],encoding='utf-8')
    #origin= pd.read_csv('/data2/wzf/dataset/disaster_new_20200823.csv',usecols=['label','content'],encoding='utf-8')
    label = pd.read_csv('/data2/wzf/dataset/disaster_new_20200828.csv',usecols=['label'],encoding='utf-8',)
    content = pd.read_csv('/data2/wzf/dataset/disaster_new_20200828.csv',usecols=['content'],encoding='utf-8')
    #origin= pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['label','content'],encoding='utf-8')
    #origin= pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['label','content'],encoding='utf-8')
    #label = pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['label'],encoding='utf-8',)
    #content = pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['content'],encoding='utf-8')
    X_training, X_test, y_training, y_test = train_test_split(content,label , test_size=0.3,random_state=i,stratify=label)
    X_train, X_valid, y_train, y_valid = train_test_split(X_training,y_training , test_size=0.3,random_state=i,stratify=y_training)
    #print(X_train.columns)
    print(X_test.columns)
    print(X_test['content'])
    print(X_test.shape)
    print(X_train.shape)
    print(X_valid.shape)
    #print(type(y_test))
    validation_set=pd.concat([y_valid,X_valid], axis=1)
    training_set=pd.concat([y_train,X_train], axis=1)
    test_set=pd.concat([y_test,X_test], axis=1)
    #print(df_inner)
    #validation_set = pd.DataFrame([y_test ;X_test], columns=['label','content'])
    validation_set.to_csv("/data2/wzf/dataset/validation_set.csv",index=False,encoding='utf-8')
    training_set.to_csv("/data2/wzf/dataset/training_set.csv",index=False,encoding='utf-8')
    test_set.to_csv("/data2/wzf/dataset/test_set.csv",index=False,encoding='utf-8')
    origin.to_csv("/data2/wzf/dataset/disasters.csv",index=False,encoding='utf-8')


    #data=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    #frame = pd.DataFrame(data, columns=['ball', 'pen', 'pencil', 'paper'])
    #print(frame)
    #frame.to_csv("./testcsv_04.csv")""))''''''''])]]]]]''"")''"")''"")''"")'''']]))])])]))))))))'''']'')'''']'')'''''']'')))
if __name__=="__main__":
    a(0)