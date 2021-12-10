
from pathlib import Path
import numpy as np
from myDataset import myDataset
from sklearn import multiclass,svm
import sklearn.metrics as metrics
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from sklearn.metrics import classification_report
import torch
import  random
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import gensim
from model import FeedfowardTextClassifier
import  matplotlib.pyplot as plt
# PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
rand=random.randint(0,100)
print(rand)
#rand=29
def split_train_valid_test(corpus, valid_ratio=0.21, test_ratio=0.3):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(corpus, lengths=[train_length, valid_length, test_length],)
def train_epoch(model, optimizer, train_loader):
    model.train()
    total_loss, total = 0, 0
    for  seq, bow, tfidf, target, text,lda in train_loader:

        inputs=lda
        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)

        # Compute loss

        loss = criterion(output, target)

        # Perform gradient descent, backwards pass
        loss.backward()

        # Take a step in the right direction
        optimizer.step()
        scheduler.step()

        # Record metrics
        total_loss += loss.item()
        total += len(target)

    return total_loss / total

def validate_epoch(model, valid_loader, input_type='lda'):
    model.eval()
    total_loss, total = 0, 0
    with torch.no_grad():
        for seq, bow, tfidf, target, text,lda in valid_loader:
            if input_type == 'bow':
                inputs = bow
            if input_type == 'tfidf':
                inputs = tfidf
            if input_type =='lda':
                inputs=lda
            # Forward pass
            output = model(inputs)

            # Calculate how wrong the model is
            loss = criterion(output, target)

            # Record metrics
            total_loss += loss.item()
            total += len(target)
    return total_loss / total
DATA_PATH = '../dataset/disaster_news_dataset.csv'

if not Path(DATA_PATH).is_file():
    print("file not exit")
    exit(-1)

dataset = myDataset(DATA_PATH)
common_texts=np.array(dataset.df['tokens'])


labels=7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=30

train_dataset, valid_dataset, test_dataset = split_train_valid_test(
    dataset)

def collate(batch):
    seq = [item[0] for item in batch]
    bow = [item[1] for item in batch]
    tfidf = [item[2] for item in batch]
    target = torch.LongTensor([item[3] for item in batch])
    text = [item[4] for item in batch]
    lda=[item[5] for item in batch]
    return seq, bow, tfidf, target, text,lda

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
LR_model = FeedfowardTextClassifier(
    vocab_size=7,
    num_labels=labels,
    device=device,
    batch_size=BATCH_SIZE,
    hidden1=100,
    hidden2=50,
)
# training
LEARNING_RATE = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, LR_model.parameters()),
    lr=LEARNING_RATE,
)
scheduler = CosineAnnealingLR(optimizer, 1)
n_epochs = 0
train_losses, valid_losses = [], []
for epoch in range(1000):
    train_loss = train_epoch(LR_model, optimizer, train_loader)
    valid_loss = validate_epoch(LR_model, valid_loader)

    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',
    )

    # Early stopping if the current valid_loss is greater than the last three valid losses
    if len(valid_losses) > 500 and all(valid_loss >= loss for loss in valid_losses[-5:]) :
        print('Stopping early')
        break

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    n_epochs += 1
epoch_ticks = range(1, n_epochs + 1)
plt.plot(epoch_ticks, train_losses)
plt.plot(epoch_ticks, valid_losses)
plt.legend(['Train Loss', 'Valid Loss'])
plt.title('Losses')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.xticks(epoch_ticks)
#plt.show()

LR_model.eval()
test_accuracy, n_examples = 0, 0
y_true, y_pred = [], []
input_type = 'lda'

with torch.no_grad():
    for seq, bow, tfidf, target, text,lda in test_loader:
        inputs = lda
        probs = LR_model(inputs)
        probs = probs.detach().cpu().numpy()
        predictions = np.argmax(probs, axis=-1)
        target = target.cpu().numpy()
        y_true.extend(predictions)
        y_pred.extend(target)

print(classification_report(y_true, y_pred))

LR_model.eval()
with torch.no_grad():
    seq, bow, tfidf, target, text,lda = dataset[6]
    target = target
    inputs = lda
    probs = LR_model(inputs)
    probs = probs.detach().cpu().numpy()
    print(probs)
    predictions = np.argmax(probs, axis=-1)

    print(target,predictions)





