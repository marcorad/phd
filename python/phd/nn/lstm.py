import torch
from torch import nn
from torch import Tensor
from typing import List
from torch import optim
import numpy as np
import random
from math import floor

from sklearn.preprocessing import LabelEncoder

def calculate_conv_out_size(in_size, ker, stride, pad = 0, dil = 1):
    return [floor( (x + 2 * pad - dil * (ker - 1) -1)/stride + 1 ) for x in in_size]

class CNN(nn.Module):
    def __init__(self, im_size, in_channels, kernel_sizes: List[int], out_channels: List[int], conv_strides: List[int], pool_sizes: List[int], pool_strides: List[int], fully_connected_sizes: List[int]) -> None:
        super(CNN, self).__init__()
        self.conv_layers = []
        self.pool_layers = []
        prev_size = im_size
        for i in range(len(kernel_sizes)):
            k = kernel_sizes[i]
            c = out_channels[i]
            s = conv_strides[i]
            prev_channels = in_channels if i == 0 else out_channels[i-1]
            self.conv_layers.append(nn.Conv2d(in_channels=prev_channels, out_channels=c, kernel_size=k, stride=s))
            prev_size = calculate_conv_out_size(prev_size, k, s)
            
            pk = pool_sizes[i]
            ps = pool_strides[i]
            self.pool_layers.append(nn.MaxPool2d(kernel_size=pk, stride=ps))
            prev_size = calculate_conv_out_size(prev_size, pk, ps)
        self.num_conv_feats = prev_size[0] * prev_size[1] * out_channels[-1]
        
        self.lin_layers = []
        for i in range(len(fully_connected_sizes)):
            in_s = self.num_conv_feats if i == 0 else fully_connected_sizes[i-1]
            self.lin_layers.append(nn.Linear(in_features=in_s, out_features=fully_connected_sizes[i]))

        self.out_features = fully_connected_sizes[-1]

    def forward(self, x: Tensor):
        y = x[:, None, :, :]
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            y = conv(y)
            y = pool(y)
        y = torch.flatten(y, start_dim=1)
        for lin in self.lin_layers:
            y = lin(y)
        return y

class CNNLSTM(nn.Module):    
    def __init__(self, cnn: CNN, lstm_size, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = cnn
        self.lstm = nn.LSTM(input_size=cnn.out_features, hidden_size=lstm_size, num_layers=2, bidirectional=True)
        self.out = nn.Linear(in_features=lstm_size*2, out_features=num_classes)
        self.softmax = nn.Softmax(dim=0) 
        
    def forward(self, x):
        y = self.cnn(x)
        y, _ = self.lstm(y)
        y = self.out(y[-1, :])
        return self.softmax(y)
    

class BasicLSTM(nn.Module):
    
    def __init__(self, input_size, latent_size, lstm_size,  num_classes, dropout=0.0):
        super(BasicLSTM, self).__init__()
        self.lin_latent = nn.Linear(input_size, latent_size)
        self.bn = nn.BatchNorm1d(latent_size)
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=lstm_size, num_layers=1, batch_first=True)
        self.lin_1 = nn.Linear(lstm_size, lstm_size*2)
        self.lin_out = nn.Linear(2*lstm_size, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.soft_max = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p = dropout)
        
        
        
    def forward(self, x):
        #input of (Ntime, Nfeat)
        latent_out = self.bn(self.lin_latent(x))#compress each time sample to latent space, with dropout
        self.latent_out = latent_out #save the latent out for L1 regularisation later
        lstm_out, _ = self.lstm(self.dropout(latent_out), None) #run LSTM on time samples with 0 initial state
        d_out = self.dropout(lstm_out[-1, :]) #pass the last time step with a dropout layer
        lin_1 = self.dropout(self.tanh(self.lin_1(d_out))) #hidden classification layer
        lin_out = self.lin_out(lin_1) #output classification layer
        return self.soft_max(lin_out) #probability softmax
    
class ModelTrainer():
    def __init__(self, model: BasicLSTM, dev = 'cuda', reg_norm = 1) -> None:
        self.model = model    
        self.device = torch.device(dev)
        self.model.to(self.device)
        self.reg_norm = reg_norm
        self.training_history = []
        
        
    def compute_loss(self, y_pred, y_true, lambda_ = 1e-6):
        # latent_out = self.model.latent_out
        # l1_reg = torch.mean(torch.linalg.vector_norm(latent_out, self.reg_norm , dim = 1)) * lambda_ #L1 norm over features and average
        loss = self.loss_fn(y_pred, y_true)
        return loss

    def train_model(self, X_train: List[np.ndarray], y_train: List[str], X_test: List[np.ndarray], y_test: List[str], n_epochs = 1000, batch_size = 128, lr = 1e-2, lambda_ = 1e-6):
        
        opt = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.le = LabelEncoder() 
        self.le.fit(y_train)   
        
        y_test_torch = torch.from_numpy(self.le.transform(y_test))    
        y_test_torch = nn.functional.one_hot(y_test_torch).type(torch.float32).to(self.device)    
        
        for epoch in range(n_epochs):            
            #shuffle the training data randomly
            c = list(zip(X_train, y_train))
            random.shuffle(c)
            X_train, y_train = zip(*c)
            
            y_train_torch = torch.from_numpy(self.le.transform(y_train))    
            y_train_torch = nn.functional.one_hot(y_train_torch).type(torch.float32).to(self.device)  
            
            self.model.train()
            losses = []
            i = 0
            while i < len(X_train):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    for n in range(batch_size):
                        train_index = i + n
                        if train_index >= len(X_train):
                            i = len(X_train)
                            break
                        x = torch.from_numpy(X_train[train_index].astype(np.float32)).to(self.device)  
                        y = y_train_torch[train_index, :]
                        t = 0
                        T = 5
                        y_pred = self.model(x)
                        loss = self.compute_loss(y_pred, y, lambda_)
                        losses.append(loss.item())
                        loss /= batch_size #account for gradient accumulation
                        loss.backward()
                    i += batch_size
                    opt.step()
            train_loss = np.mean(losses)
            #validate the accuracy
            print(f'{epoch=}: Average loss {train_loss}')
            self.model.eval()
            N_val = len(X_test)
            n_corr = 0
            val_losses = []
            with torch.set_grad_enabled(False):
                for n in random.choices(list(range(len(X_test))), k=N_val):
                    x = torch.from_numpy(X_test[n].astype(np.float32)).to(self.device)  
                    y = y_test_torch[n, :]
                    y_pred = self.model(x)
                    loss = self.compute_loss(y_pred, y, lambda_)
                    y_pred = torch.argmax(y_pred)
                    y_true = torch.argmax(y)
                    val_losses.append(loss.item())
                    if y_pred == y_true: n_corr += 1
            val_loss = np.mean(val_losses)   
            
            print(f'Validation loss: {val_loss} ({n_corr/N_val*100}% accuracy)')
            self.training_history.append([train_loss, val_loss])
                    
                
                
                
        
        