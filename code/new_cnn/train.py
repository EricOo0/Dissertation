import os
import sys
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.content, batch.label
            feature.t_(), target.data.sub_(1)# t_()原地进行转置  sub_(1)把target中的每个值都减1

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.content, batch.label
        feature.t_(), target.data.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
        #print(torch.max(logits, 1)[1].view(target.size()).data)
      
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy
def eval_test(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    predict_result=[]
    accutal_result=[]
    for batch in data_iter:
        feature, target = batch.content, batch.label
        feature.t_(), target.data.sub_(1)
        
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
        predict_result=predict_result+torch.max(logits, 1)[1].view(target.size()).data.tolist()
        accutal_result=accutal_result+target.data.tolist()
       # precission and recall 
       #
    print(len(predict_result))
    print(predict_result)

    print(len(accutal_result))
    print(accutal_result)
    # 计算总的精度
    acc = accuracy_score(accutal_result, predict_result)
    p= precision_score(accutal_result, predict_result, average='weighted')#参数average有5个选项：{‘micro’微平均, ‘macro’宏平均, ‘samples’, ‘weighted’, ‘binary’}
    r = recall_score(accutal_result, predict_result, average='weighted')
    f1score = f1_score(accutal_result, predict_result, average='weighted')
    print("acc{:.6f}, p{:.6f},recall{:.6f},f1score{:.6f}".format(acc,p,r,f1score))
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return acc,p,r,f1score
    

def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
