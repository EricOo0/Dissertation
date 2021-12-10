
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
import gensim
rand=random.randint(0,20)
print(rand)
def split_train_valid_test(corpus, valid_ratio=0.21, test_ratio=0.3):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(corpus, lengths=[train_length, valid_length, test_length],)

DATA_PATH = '../dataset/disaster_news_dataset.csv'

if not Path(DATA_PATH).is_file():
    print("file not exit")
    exit(-1)

dataset = myDataset(DATA_PATH)
common_texts=np.array(dataset.df['tokens'])
np.array(common_texts)
#把文章转换成list
dictionary=Dictionary(common_texts)
#把文本转换成词袋的形式  id：freq
corpus=[dictionary.doc2bow(text) for text in common_texts]
lda=LdaModel(corpus,id2word=dictionary,num_topics=7,iterations = 600,chunksize = 30,passes = 30)
corpus_topic=lda[corpus]
numpy_matrix = gensim.matutils.corpus2dense(corpus_topic,num_terms = 7)
numpy_matrix = numpy_matrix.T
label=np.array(dataset.targets)
len = numpy_matrix.shape[0]
dic =[]
for i in range(len):
    dic.append({label[i]:numpy_matrix[i][:]})
random.shuffle(dic)
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for t in dic[:-480]:
    fff=list(list(t.values())[0])
    b=type(fff)
    X_train.append(list(list(t.values())[0]))
    y_train.append(list( t.keys())[0])
for t in dic[-480:]:
    X_test.append(list(list(t.values())[0]))
    y_test.append( list(t.keys())[0])


model = multiclass.OneVsRestClassifier(svm.SVC( probability=True,))
print("[INFO] Successfully initialize a new model !")
print("[INFO] Training the model…… ")
clt = model.fit(X_train, y_train)
print("[INFO] Model training completed !")


y_test_pred = clt.predict(X_test)
ov_acc = metrics.accuracy_score(y_test_pred, y_test)
print("overall accuracy: %f" % (ov_acc))
print("===========================================")
acc_for_each_class = metrics.precision_score(y_test, y_test_pred, average=None)
print("acc_for_each_class:\n", acc_for_each_class)
print("===========================================")
avg_acc = np.mean(acc_for_each_class)
print("average accuracy:%f" % (avg_acc))
print(classification_report(y_test, y_test_pred,digits=5))





