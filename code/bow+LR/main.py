import random
import re # regular expression
from collections import Counter
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from IPython.core.display import display, HTML
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF
from sklearn.metrics import classification_report
from tqdm import tqdm, tqdm_notebook # show progress bar

# PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split




from model import LogisticRegressionClassifier
from myDataset import myDataset

DATA_PATH = '../dataset/disaster_news_dataset.csv'
if not Path(DATA_PATH).is_file():
    print("file not exit")
    exit(-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(DATA_PATH)
df.sample(5)

print('Number of records:', len(df), '\n')
print('Number of drought :', len(df[df.label == 'drought']))
print('Number of earthquake :', len(df[df.label == 'earthquake']), '\n')

print('Example 1 :')
print(df.loc[55,].content, '\n')
print('Example 2 :')
print(df.loc[100,].content, '\n')

def split_train_valid_test(corpus, valid_ratio=0.21, test_ratio=0.3):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(corpus, lengths=[train_length, valid_length, test_length],)

def train_epoch(model, optimizer, train_loader, input_type='bow'):
    model.train()
    total_loss, total = 0, 0
    for seq, bow, tfidf, target, text in train_loader:
        if input_type == 'bow':
            inputs = bow
        if input_type == 'tfidf':
            inputs = tfidf

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

def validate_epoch(model, valid_loader, input_type='bow'):
    model.eval()
    total_loss, total = 0, 0
    with torch.no_grad():
        for seq, bow, tfidf, target, text in valid_loader:
            if input_type == 'bow':
                inputs = bow
            if input_type == 'tfidf':
                inputs = tfidf

            # Forward pass
            output = model(inputs)

            # Calculate how wrong the model is
            loss = criterion(output, target)

            # Record metrics
            total_loss += loss.item()
            total += len(target)

    return total_loss / total
dataset = myDataset(DATA_PATH)

print('Number of records:', len(dataset), '\n')
random_idx = random.randint(0,len(dataset)-1)
print('index:', random_idx, '\n')
sample_seq, bow_vector, tfidf_vector, sample_target, sample_text = dataset[random_idx]
print(sample_text, '\n')
print(sample_seq, '\n')
print('BoW vector size:', len(bow_vector), '\n')
print('TF-IDF vector size:', len(tfidf_vector), '\n')
print('Sentiment:', sample_target, '\n')

train_dataset, valid_dataset, test_dataset = split_train_valid_test(
    dataset)
BATCH_SIZE = 30

def collate(batch):
    seq = [item[0] for item in batch]
    bow = [item[1] for item in batch]
    tfidf = [item[2] for item in batch]
    target = torch.LongTensor([item[3] for item in batch])
    text = [item[4] for item in batch]
    return seq, bow, tfidf, target, text

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

len(train_dataset), len(valid_dataset), len(test_dataset)
labels=len(np.unique(dataset.targets))
print(labels);
bow_model = LogisticRegressionClassifier(
    vocab_size=len(dataset.token2idx),
    num_labels=labels,
    device=device,
    batch_size=BATCH_SIZE,
)
# training
LEARNING_RATE = 2e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, bow_model.parameters()),
    lr=LEARNING_RATE,
)
scheduler = CosineAnnealingLR(optimizer, 1)
n_epochs = 0
train_losses, valid_losses = [], []
for epoch in range(50):
    train_loss = train_epoch(bow_model, optimizer, train_loader, input_type='tfidf')
    valid_loss = validate_epoch(bow_model, valid_loader, input_type='tfidf')

    tqdm.write(
        f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',
    )

    # Early stopping if the current valid_loss is greater than the last three valid losses
    if len(valid_losses) > 2 and all(valid_loss >= loss for loss in valid_losses[-5:]) :
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

bow_model.eval()
test_accuracy, n_examples = 0, 0
y_true, y_pred = [], []

with torch.no_grad():
    for seq, bow, tfidf, target, text in test_loader:
        inputs = bow
        probs = bow_model(inputs)
        probs = probs.detach().cpu().numpy()
        predictions = np.argmax(probs, axis=-1)
        target = target.cpu().numpy()
        y_true.extend(predictions)
        y_pred.extend(target)

print(classification_report(y_true, y_pred))

bow_model.eval()
with torch.no_grad():
    seq, bow, tfidf, target, text = dataset[random_idx]
    target = target
    inputs = bow
    probs = bow_model(inputs)
    probs = probs.detach().cpu().numpy()
    print(probs)
    predictions = np.argmax(probs, axis=-1)

    print(target,predictions)