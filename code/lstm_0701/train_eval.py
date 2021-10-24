#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:53:31 2019

@author: jjg
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
import numpy as np
import random
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

#from torchtext import data

     
def training(train_iter, dev_iter, model, args, device):
    l2 = args.l2
    model.to(device)  #move model to device before constructing optimizer for it.
    if not args.static and not args.sim_static:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2)
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=l2)

    total_step = len(train_iter)
    train_accs = []
    dev_accs = []
    best_acc = 0
    t0 = time()
    for epoch in range(1, args.epochs + 1):
        model.train()  #training mode, we should reset it to training mode in each epoch
        for i, batch in enumerate(train_iter):
            texts, labels = batch.content.to(device), batch.label.to(device) - 1
            outputs = model(texts)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #Clears the gradients

            #Visualization of the train process
            if i % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch, args.epochs, i+1, total_step, loss.item()))
    
        #in each epoch we call eval(), switch to evaluation mode
        train_accs.append(evaluating(train_iter, model, device))
        dev_acc = evaluating(dev_iter, model, device)
        dev_accs.append(dev_acc)
        if dev_acc > best_acc:
             best_acc = dev_acc
             best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    t1 = time()
    print('training time: %.2f' % (t1 - t0))
    show_training(train_accs, dev_accs)
    return model


def evaluating(data_iter, model, device):
    model.to(device)#把模型加载到设备上
    model.eval()  #evaluation mode
    with torch.no_grad():#不计算梯度
        correct, avg_loss = 0, 0
        for batch in data_iter:
            texts, labels = batch.content.to(device), batch.label.to(device) - 1

            outputs = model(texts)
            predicted = torch.max(outputs.data, 1)[1]
            loss = F.cross_entropy(outputs, labels, reduction='mean')

            avg_loss += loss.item()
            correct += (predicted == labels).sum()

        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = correct.item() / size
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 100*accuracy, correct, size))
        return accuracy
def eval_final(data_iter, model, device):
    model.to(device)
    model.eval()
    corrects, avg_loss = 0, 0
    predict_result=[]
    accutal_result=[]

    for batch in data_iter:
        feature, target = batch.content.to(device), batch.label.to(device)-1
        #feature.t_(), target.data.sub_(1)
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        predict=torch.max(logits.data, 1)[1]
        avg_loss += loss.item()
        corrects += (predict == target).sum()
        predict_result=predict_result+predict.tolist()
        accutal_result=accutal_result+target.tolist()
    # precission and recall 
    #
    print("predict:")
    print(predict_result)
    print("actual:")
    print(accutal_result)
    # 计算总的精度
    acc = accuracy_score(accutal_result, predict_result)
    p= precision_score(accutal_result, predict_result, average='weighted')#参数average有5个选项：{‘micro’微平均, ‘macro’宏平均, ‘samples’, ‘weighted’, ‘binary’}
    r = recall_score(accutal_result, predict_result, average='weighted')
    f1score = f1_score(accutal_result, predict_result, average='weighted')
    #print("acc:{:.6f}, p:{:.6f},recall:{:.6f},f1score:{:.6f}".format(acc,p,r,f1score))
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

    return acc,p,r,f1score
   
def show_training(train_accs, dev_accs):
    #plot train acc and validation acc
    plt.ion()
    plt.figure()
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.title('train_acc and dev_acc vs epochs')
    plt.tight_layout()
    #plt.xticks(range(0,args.epochs),range(1,args.epochs+1))
    plt.plot(train_accs, label='train_acc')
    plt.plot(dev_accs, label='dev_acc')
    plt.legend()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

 
#def training_big(train, chunkSize, numChunk, dev_iter, model, args, device):
#    l2 = args.l2
#    model.to(device)  #move model to device before constructing optimizer for it.
#    if not args.static and not args.sim_static:
#        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2)
#    else:
#        optimizer = optim.Adam(
#            filter(lambda p: p.requires_grad, model.parameters()),
#            lr=args.lr,
#            weight_decay=l2)
#
#    train_accs = []
#    dev_accs = []
#    best_acc = 0
#    t0 = time()
#    for epoch in range(1, args.epochs + 1):
#        # chunk-by-chunk training
#        for chunk in range(numChunk):
#            train.initial()
#            train_iter = data.BucketIterator(dataset=train, batch_size=args.batch_size, shuffle=True, sort_within_batch=False, repeat=False)
#            total_step = len(train_iter)
#            model.train()  #training mode, we should reset it to training mode in each epoch
#            for i, batch in enumerate(train_iter):
#                texts, labels = batch.text.to(device), batch.label.to(device) - 1
#                optimizer.zero_grad()
#                outputs = model(texts)
#    
#                loss = F.cross_entropy(outputs, labels)
#                loss.backward()
#                optimizer.step()
#    
#                #Visualization of the train process
#                if i % 100 == 0:
#                    print('Epoch [{}/{}], Chunk [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
#                        epoch, args.epochs, chunk+1, numChunk, i+1, total_step, loss.item()))
#        
#        #in each epoch we call eval(), switch to evaluation mode
#        train_accs.append(evaluating(train_iter, model, device))
#        dev_acc = evaluating(dev_iter, model, device)
#        dev_accs.append(dev_acc)
#        if dev_acc > best_acc:
#             best_acc = dev_acc
#             best_model_wts = copy.deepcopy(model.state_dict())
#    
#    model.load_state_dict(best_model_wts)
#    t1 = time()
#    print('training time: %.2f' % (t1 - t0))
#    show_training(train_accs, dev_accs)
#    return model
