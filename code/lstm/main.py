#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:53:44 2019

@author: jjg
"""
import nltk
nltk.download('punkt')
import torch
import os
import numpy as np
from torchtext.legacy import data
from torchtext.vocab import Vectors
from nltk.tokenize import word_tokenize
from models import TextCNN, LSTM, GRU, BiLSTM_LSTM
#u can ignore sims
from sims import Hybrid_CNN, SimCNN, SimLSTM, SimLSTM1, SimLSTM2, SimLSTM3,\
SimLSTM4, SimLSTM5, SimLSTM6, SimLSTM7, SimLSTM8, SimLSTM9, SimLSTM10, \
SimLSTM11, SimAttn, SimAttn1, SimAttn2, SimAttn3, SimAttnPE1, SimCnnPe, SimAttnX

from settings import Settings
from train_eval import training, evaluating,eval_final
import sys
sys.path.append('../dataset/')
import data_extraction as self_data

#available models, u can use the first 3 models: TextCNN, LSTM, GRU
models = {
    'TextCNN': TextCNN,
    'LSTM': LSTM,
    'GRU': GRU,
    'SimCNN': SimCNN,
    'SimLSTM': SimLSTM,
    'Hybrid': Hybrid_CNN,
    'SimLSTM1': SimLSTM1,
    'SimLSTM2': SimLSTM2,
    'SimLSTM3': SimLSTM3,
    'SimLSTM4': SimLSTM4,
    'SimLSTM5': SimLSTM5,
    'SimLSTM6': SimLSTM6,
    'SimLSTM7': SimLSTM7,
    'SimLSTM8': SimLSTM8,
    'BiLSTM_LSTM': BiLSTM_LSTM,
    'SimLSTM9': SimLSTM9,
    'SimLSTM10': SimLSTM10,
    'SimLSTM11': SimLSTM11,
    'SimAttn': SimAttn,
    'SimAttn1': SimAttn1,
    'SimAttn2': SimAttn2,
    'SimAttnPE1': SimAttnPE1,
    'SimAttn3': SimAttn3,
    'SimCnnPe': SimCnnPe,
    'SimAttnX': SimAttnX
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '../dataset/'
vector_path = '../'
vector_path = os.path.join(vector_path, 'vector')
cache = os.path.join(vector_path, 'vector_cache')
#vector_path = '../dataset/'
#dataset_path = os.path.join(data_path, 'text_classifiacation', 'dbpedia')
dataset_path=data_path
# class_vecs is used for models in sims.py, if u don't need, just set use_sims to False
use_sims = False
if use_sims:
    class_vecs = torch.from_numpy(
        np.load(os.path.join(dataset_path, 'bow_based_label_embedding.npy')))
else:
    class_vecs = torch.randn(14, 300)  #14/dbpedia, 10/yahoo
length = 100  #100/dbpedia; 300/yahoo

if __name__ == '__main__':
    accuracy=[]
    precise=[]
    recall=[]
    f1=[]
    for i in range(10):
        print("number i :{:d}".format(i))
        self_data.a(i)
        #set batch_first=True
        TEXT = data.Field(
            sequential=True,
            tokenize=word_tokenize,
            lower=True,
            fix_length=None,  #according to the dataset
            batch_first=True)

        #LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
        LABEL = data.Field(sequential=False)
        fields = {'label': LABEL, 'content': TEXT}

        #use word2vec embeddings(change GoogleNews-vectors-negative300.bin to googlenews.txt)
        #or use Glove embeddings
        #TEXT.build_vocab(train, vectors="glove.840B.300d") download the word embedding file
        
        if not os.path.exists(cache):
            os.mkdir(cache)
        
        #vectors = Vectors(
            
            # name=os.path.join(vector_path,'googlenews.txt'),
        #    name='/data2/wzf/dataset/word2vec.txt',
            #name=os.path.join(vector_path, 'glove.840B.300d.txt'),#torch.Size([2196017, 300])
         #   cache=cache)

        #load data set
        train, dev, test = data.TabularDataset.splits(
            path=dataset_path,
            train='training_set.csv',
            validation='validation_set.csv',
            test='test_set.csv',
            format='csv',
            fields=[('label', LABEL), ('content', TEXT)],
            skip_header=True)

        #construct the vocab, filter low frequency words if needed
        #TEXT.build_vocab(train, min_freq=2, vectors=vectors)
        #LABEL.build_vocab(train, min_freq=2, vectors=vectors)
        #print("#pretrained")
        TEXT.build_vocab(train, min_freq=2)
        LABEL.build_vocab(train, min_freq=2)

        print(len(LABEL.vocab))
        print(LABEL.vocab.itos[0:-1])
        #del vectors  #del vectors to save space
        #print(len(LABEL.vocab))


        #settings, see settings.py for detail
        label_vecs = class_vecs.unsqueeze(1).unsqueeze(2)  #u can ignore this
        args = Settings(
            #TEXT.vocab.vectors,  #pre-trained word embeddings
            len(TEXT.vocab),
            label_vecs,
            L=length,
            Dim=300,             #embedding dimension
            num_class=len(LABEL.vocab),        #14/dbpedia, 10/yahoo
            Cout=128,            #kernel numbers
            kernel_size=[3, 4, 5],  #different kernel size
            dropout=0.5,
            num_epochs=32,
            lr=0.001,
            weight_decay=0,
            #static=True,            #update the embeddings or not
            static=False,
            sim_static=True,    #only used for sims.py: update label_vec_kernel or not
            batch_size=8,
            batch_normalization=False,
            hidden_size=256,    #100,128,256...
            rnn_layers=2,
            bidirectional=True)  #birnn/rnn

        #construct dataset iterator
        train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test),
            sort_key=lambda x: len(x.content),
            batch_sizes=[args.batch_size] * 3,
        )
        classifier = 'LSTM'
        if args.static:
            print('static %s(without updating embeddings):' % classifier)
        else:
            print('non-static %s(update embeddings):' % classifier)

        model = models[classifier](args)
        training(train_iter, dev_iter, model, args, device)

        #finally evaluate the test set
        test_acc = evaluating(test_iter, model, device)
        print('test acc: %.1f%%' % (test_acc * 100)) #98.8/dbpedia

        acc,p,r,f1score=eval_final(test_iter,model,device)
        accuracy.append(acc)
        precise.append(p)
        recall.append(r)
        f1.append(f1score)      
    print(accuracy)
    avg_acc=np.mean(accuracy)
    print(recall)
    avg_recall=np.mean(recall)
    print(precise)
    avg_precise=np.mean(precise)
    print(f1)
    avg_f1=np.mean(f1)
    print("avg_acc:{:.6f}, avg_p:{:.6f},avg_eecall:{:.6f},avg_f1score:{:.6f}".format(avg_acc,avg_precise,avg_recall,avg_f1)) 

