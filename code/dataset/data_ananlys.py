import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import jieba
df = pd.read_csv("./disaster_new_20200828.csv")
arr = []
label =[]
regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
for i in df.index:
    text = regex.sub(' ', df.content[i])
    tmp=[word for word in jieba.cut(text) if word.strip()]
    arr.append(len(tmp))
average = np.mean(arr)
var = np.var(arr)
std = np.std(arr)
max = np.max(arr)
min = np.min(arr)
bin=np.unique(arr)
len_count=pd.value_counts(arr, sort=False)
len_label=pd.unique(arr)
label=df.label
count=pd.value_counts(label, sort=False)
label=pd.unique(label)
print(average, var, std, max, min, len(bin))
