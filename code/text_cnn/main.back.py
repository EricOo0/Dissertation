import argparse
import torch
import torchtext.data as data
from torchtext.vocab import Vectors

import model
import train
import dataset

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=32, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=True, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='../vector/glove.840B.300d.txt',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='../vector/vector_cache', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
args = parser.parse_args()

print(args.static)
def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    #vectors = Vectors(）
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset,test_dataset = dataset.get_dataset('', text_field, label_field)
   
    #是否加载预训练模型
    #建立label和content的词典
    #args.static=False
    print(args.static)
    print(args.non_static)
    if args.static and args.pretrained_name and args.pretrained_path:
        
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset,test_dataset, vectors=vectors)
        print("#pretrained")
    else:
        print("#not pretrained")
        text_field.build_vocab(train_dataset, dev_dataset,test_dataset)
    label_field.build_vocab(train_dataset, dev_dataset, test_dataset)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset,dev_dataset,test_dataset), sort_key=lambda x: len(x.content),
        batch_sizes=(args.batch_size,args.batch_size,args.batch_size), device=-1)
  #  train_iter, dev_iter = data.Iterator.splits(
  #      (train_dataset, dev_dataset),
   #     batch_sizes=(args.batch_size, len(dev_dataset)),
    #    sort_key=lambda x: len(x.content),
    #    **kwargs)
    #test_iter = data.Iterator(test_dataset,batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
    return train_iter, dev_iter, test_iter

'''
#k-fold
for i in range(10):
    
    train_iter, dev_iter,test_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
    k-fold.appen()
'''
print('Loading data...')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter,test_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
args.vocabulary_size = len(text_field.vocab)

#词汇表长度 会加入一个标签unk
print("label_len:%d" %len(label_field.vocab))
print(label_field.vocab.itos[0:6])#itos 按照下标的顺序返回每一个单词. stoi 返回每一个单词与其对应的下标
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True      
print("embedding_dim")
print(args.embedding_dim)
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

text_cnn = model.TextCNN(args)
if args.snapshot:
    print('\nLoading model from {}...\n'.format(args.snapshot))
    text_cnn.load_state_dict(torch.load(args.snapshot))
if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()


try:
    train.train(train_iter, dev_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')
if args.predict is not None:
    label = train.predict(args.predict, text_cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    acc=train.eval(test_iter, text_cnn, args)
train.eval_test(test_iter,text_cnn,args)
#python main.py -snapshot ./snapshot/best_steps_100.pt -predict "typhone is comming,if you come home ,you will safe!" 