
from pathlib import Path
import numpy as np
from myDataset import myDataset
from sklearn import multiclass,svm
import sklearn.metrics as metrics

from torch.utils.data.dataset import random_split
from sklearn.metrics import classification_report
import torch
import  random
rand=random.randint(0,20)
print(rand)
def split_train_valid_test(corpus, valid_ratio=0.21, test_ratio=0.3):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(corpus, lengths=[train_length, valid_length, test_length],generator=torch.Generator().manual_seed(rand))

DATA_PATH = '../dataset/disaster_news_dataset.csv'

if not Path(DATA_PATH).is_file():
    print("file not exit")
    exit(-1)

dataset = myDataset(DATA_PATH)
train_dataset, valid_dataset, test_dataset = split_train_valid_test(
   dataset,0,0.3)
train_dataset
tmp = [dataset[i]for i in train_dataset.indices]
random.shuffle(tmp)
X_train=[i[2]for i in tmp]
y_train=[i[3]for i in tmp]

X_test=[dataset[i][2]for i in test_dataset.indices]
y_test=[dataset[i][3]for i in test_dataset.indices]

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
print("average accuracy:%f" %(avg_acc))
print(classification_report(y_test, y_test_pred,digits=5))
a=classification_report(y_test, y_test_pred,digits=5)
