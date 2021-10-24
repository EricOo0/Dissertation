import re
from torchtext import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)
#正则表达式
#u4e00-\u9fa5a匹配所有作文
#A-Za-z0-9 匹配所有数字字母

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)#空格分割
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev ,test= data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train='../dataset/training_set.csv', validation='../dataset/validation_set.csv',test='../dataset/test_set.csv',
        fields=[
            ('label', label_field),
            ('content', text_field)
        ]
    )
    return train, dev,test
