from pathlib import Path
import numpy as np
from myDataset import myDataset
from sklearn import multiclass,svm
import sklearn.metrics as metrics

from torch.utils.data.dataset import random_split
from sklearn.metrics import classification_report
import torch
import  random
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import matplotlib as plt
import os
import pandas as pd
import gc
import lda
DATA_PATH = '../dataset/disaster_news_dataset.csv'

if not Path(DATA_PATH).is_file():
    print("file not exit")
    exit(-1)
def load_data():
    '''
    加载lda中的数据
    '''
    tmp1=pd.read_csv(DATA_PATH,usecols=['content'])
    print("train.CSV读完了~")
    tmp2=pd.DataFrame(tmp1)
    print("train.CSV转为dataframe~")
    del tmp1
    gc.collect()
    X=tmp2.values
    print("label.dataframe转为list~")
    del tmp2
    gc.collect()
    current_path = os.getcwd()
    tmp1=pd.read_csv("/Users/weizhifeng/github/Dissertation/code/dataset/test_set.csv",usecols=['content'])
    print("test.CSV读完了~")
    tmp2=pd.DataFrame(tmp1)
    print("test.CSV转为dataframe~")
    del tmp1
    gc.collect()
    Y=tmp2.values
    print("label.dataframe转为list~")
    del tmp2
    gc.collect()
    print(X.shape)
    print(X.sum())
    return X,Y
X,Y=load_data()
model=lda.LDA(n_topics=7, n_iter=800, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word=model.topic_word_  # model.components_ also works
n_top_words = 7
print('-|'*50)
doc_topic = model.doc_topic_
for i in range(7):
    predict=[0]*7
    for j in range(5000):
    #print("{} (top topic: {})".format(" ", doc_topic[i].argmax()))
        predict[doc_topic[i*5000+j].argmax()]+=1
    print("第"+str(i)+"类准确率="+str(max(predict)*1.0/5000))
print('-|'*50)
plt.plot(model.loglikelihoods_[5:])
plt.savefig('lda_test.png')
Z=model.transform(Y)
whole_right=0
for i in range(7):
    predict=[0]*7
    for j in range(5000):
    #print("{} (top topic: {})".format(" ", doc_topic[i].argmax()))
        predict[Z[i*5000+j].argmax()]+=1
    print("第"+str(i)+"类准确率="+str(max(predict)*1.0/5000))
    whole_right+=max(predict)
print("总准确率="+str(whole_right*1.0/50000))
