import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
def a(i):
    tmp=os.path.abspath(__file__)
    path=os.path.dirname(tmp)

    origin= pd.read_csv(path+'/disaster_news_dataset.csv',usecols=['label','content'],encoding='utf-8')
    #origin= pd.read_csv('/data2/wzf/dataset/disaster_new_20200823.csv',usecols=['label','content'],encoding='utf-8')
    label = pd.read_csv(path+'/disaster_news_dataset.csv',usecols=['label'],encoding='utf-8',)
    content = pd.read_csv(path+'/disaster_news_dataset.csv',usecols=['content'],encoding='utf-8')
    #origin= pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['label','content'],encoding='utf-8')
    #origin= pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['label','content'],encoding='utf-8')
    #label = pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['label'],encoding='utf-8',)
    #content = pd.read_csv('/data2/wzf/dataset/bbc_news.csv',usecols=['content'],encoding='utf-8')
    X_training, X_test, y_training, y_test = train_test_split(content,label , test_size=0.3,random_state=i,stratify=label)
    X_train, X_valid, y_train, y_valid = train_test_split(X_training,y_training , test_size=0.3,random_state=i,stratify=y_training)
    #print(X_train.columns)
    print("Test_Dataset example : ")
    print(X_test.columns)
    print(X_test['content'])
    print("Tset_Dataset       ---- len:{},shape:{}".format(X_test.shape[0],X_test.shape))
    print("Training_Dataset   ---- len:{},shape:{}".format(X_train.shape[0],X_train.shape))
    print("Validation_Dataset ---- len:{},shape:{}".format(X_valid.shape[0],X_valid.shape))
    #print(type(y_test))
    validation_set=pd.concat([y_valid,X_valid], axis=1)
    training_set=pd.concat([y_train,X_train], axis=1)
    test_set=pd.concat([y_test,X_test], axis=1)
    #print(df_inner)
    #validation_set = pd.DataFrame([y_test ;X_test], columns=['label','content'])
    validation_set.to_csv("validation_set.csv",index=False,encoding='utf-8')
    training_set.to_csv("training_set.csv",index=False,encoding='utf-8')
    test_set.to_csv("test_set.csv",index=False,encoding='utf-8')
    origin.to_csv("disasters.csv",index=False,encoding='utf-8')


    #data=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    #frame = pd.DataFrame(data, columns=['ball', 'pen', 'pencil', 'paper'])
    #print(frame)

if __name__=="__main__":
    a(0)