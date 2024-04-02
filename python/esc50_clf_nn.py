import pickle as pkl

import numpy as np
from typing import Dict
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

class LBNR(nn.Module): #linear + batch-norm + relu
    def __init__(self, in_feats, out_feats, device, bn = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats) if bn else None
        self.to(device)
        
    def forward(self, x):
        return F.tanh(self.bn(self.lin(x))) if self.bn else F.tanh(self.lin(x))

class Model(nn.Module):
    def __init__(self, in_sizes, gru_size, n_gru, out_sizes, n_classes, device) -> None:
        super().__init__()
        self.in_lin = []
        self.N_in = len(in_sizes)
        self.dropout = nn.Dropout()
        for i in range(self.N_in - 1):
            self.in_lin.append(LBNR(in_sizes[i], in_sizes[i+1], device))
        self.gru_bn = nn.BatchNorm1d(in_sizes[-1])
        self.gru = nn.GRU(in_sizes[-1], gru_size, num_layers=n_gru)
        self.N_out = len(out_sizes)
        self.out_lin = [LBNR(gru_size, out_sizes[0], device, bn=False)]
        for i in range(self.N_out):
            self.out_lin.append(LBNR(out_sizes[i], out_sizes[i+1] if i < self.N_out-1 else n_classes, device, bn=False))
        self.to(device)
            
    def forward(self, x):
        for l in self.in_lin:
            x = l(x)
        x = self.gru_bn(x)
        # x = self.dropout(x)
        _, x = self.gru(x)
        for l in self.out_lin:
            x = l(x)
        return F.softmax(x, dim=-1)
    
def train(model, X_train, y_train, X_val, y_val, lr, n_epochs):
    n_classes = len(np.unique(y_train))    
    le = LabelEncoder()
    y_train = torch.from_numpy(le.fit_transform(y_train))      
    y_val = torch.from_numpy(le.transform(y_val))
    N_train = len(X_train)      
    
    optim = torch.optim.Adam(params=model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()       
    
    
    batch_size = 64
    
    
    for n in range(n_epochs):            
        train_loss_acc = 0  
        train_idx = np.random.permutation(np.arange(N_train))
        model.train()
        print(f'Epoch {n}')
        with torch.set_grad_enabled(True):
            running_loss = 0
            batch_counter = 0
            for i in tqdm(train_idx):
                x = X_train[i]
                y = y_train[i]
                
                x = torch.from_numpy(x).cuda()
                y = F.one_hot(y, n_classes).type(torch.float32).cuda()
                # x_batch *= torch.randn_like(x_batch)*0.05 + 1
                y_pred = model(x)
                
                loss = loss_fn(y_pred[0, :], y) / batch_size
                train_loss_acc += loss.item() * batch_size
                loss.backward()
                batch_counter += 1
                
                if batch_counter == batch_size or i == train_idx[-1]:
                    batch_counter = 0                
                    optim.step()                                        
                    optim.zero_grad()
                
        #print accuracy
        model.eval()
        n_correct = 0
        N_val = len(y_val)
        loss_acc = 0
        with torch.set_grad_enabled(False):
            for x, y in zip(X_val, y_val):
                x = torch.from_numpy(x).cuda()
                y = F.one_hot(y, n_classes).type(torch.float32).cuda()
                y_pred = model(x)
                loss = loss_fn(y_pred[0, :], y)
                if torch.argmax(y) == torch.argmax(y_pred): n_correct += 1
                loss_acc += loss.item()
            
            print(f'Epoch {n} validation accuracy: {n_correct / N_val * 100: .2f} (test loss={loss_acc / N_val: .4f}, train loss={train_loss_acc / N_train: .4f}) ')
        
filename = 'data/ws-esc50-[013931972789115646, 320]-20-8-[[1, 1], [1, 1]].pkl'
with open(filename, 'rb') as file:
    X_train, y_train, X_test, y_test = pkl.load(file)
    
# y = y_train + y_test
# X = np.append(X_train, X_test, axis=0)

def remove_silence(X, thresh):
    Xr = []
    for x in X:
        e = np.std(x, axis=1)
        print((e > e.max()/thresh).sum(), e.min(), e.max())
        Xr.append(x[e > e.max()/thresh, :])
    return Xr    


mu = X_train.mean()
std = X_train.std()
X_train = (X_train - mu)/std
X_test = (X_test - mu)/std

thresh = 1.5
# X_train = remove_silence(X_train, thresh=thresh)    
# X_test = remove_silence(X_test, thresh=thresh)  

device = torch.device('cuda')
model = Model([X_train[0].shape[-1], 4096], 1024, 2, [512, 1024, 256], 50, device)

train(model, X_train, y_train, X_test, y_test, 2e-3, 100)  



        
        
    
