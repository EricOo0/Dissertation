import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, device, batch_size, vocab_size, num_labels):  # pass layer dimension
        super(LogisticRegressionClassifier, self).__init__()
        self.device = device
        self.batch_size=batch_size

        ## add affine layer
        self.lr = nn.Linear(vocab_size, num_labels)  # 相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
        self.sm = nn.Softmax()  # 相当于通过激活函数的变换

    def forward(self, x):
        batch_size = len(x)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        ## pass through affine layer & activation
        x = torch.FloatTensor(x)
        x = self.lr(x)

        return torch.softmax(x,-1)