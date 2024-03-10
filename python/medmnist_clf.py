import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle as pkl
from sklearn import metrics
import torch
from phd.dataprocessing.medmnist3d import DATASETS

import torch

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder

import torch.nn as nn



class DeepClassifier(nn.Module):
    def __init__(self, input_shape, hidden_sizes, num_classes) -> None:
        super().__init__()
        self.lin_in = nn.Linear(input_shape, hidden_sizes[0])
        self.bn_in = nn.BatchNorm1d(hidden_sizes[0])
        self.bn = []
        self.lin_hid = []
        for i in range(len(hidden_sizes)-1):
            self.lin_hid.append(nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i+1]))
            self.bn.append(nn.BatchNorm1d(hidden_sizes[i+1]))
        self.lin_out = nn.Linear(hidden_sizes[-1], num_classes) if num_classes > 2 else nn.Linear(hidden_sizes[-1], 1)
        self.soft_max = nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid()
        self.non_linearity = nn.ReLU()
        
    def forward(self, x):
        y = self.non_linearity(self.bn_in(self.lin_in(x)))
        for l, b in zip(self.lin_hid, self.bn):
            y = self.non_linearity(b(l(y)))            
        y = self.soft_max(self.lin_out(y))
        return y
    
class BalancedDataLoader:
    def __init__(self, X: torch.Tensor, y: torch.Tensor, n, sigma = 0.0, to_one_hot = False) -> None:
        self.X = X
        self.y = y
        self.labels, self.label_counts = torch.unique(y, return_counts=True)
        self.largest_class = torch.max(self.label_counts)
        self.n_classes = self.labels.shape[0]
        self.idx_counter = 0
        self.X_dict = {}
        self.idx_dict = {}
        self.n = n #number of elements of each class in each batch
        self.batch_size = n * self.n_classes
        self.to_one_hot = to_one_hot
        self.sigma = sigma
        for l in self.labels:
            self.X_dict[l] = X[self.y == l, ...]
        
        
    def _shuffle(self):
        for l, c in zip(self.labels, self.label_counts):
            self.idx_dict[l] = torch.randperm(c)
        
    def __iter__(self):        
        return self
    
    def __next__(self):
        #shuffle at the start of each iteration
        if self.idx_counter == 0: self._shuffle()
        #stop only when the largest class has finished sampling
        if self.idx_counter < self.largest_class:             
            #sample from each class, wrapping if required
            batch_x = torch.zeros((self.batch_size, *self.X.shape[1:]), dtype=self.X.dtype)
            batch_y = torch.zeros((self.batch_size, *self.y.shape[1:]), dtype=self.y.dtype)
            batch_idx = 0
            for i in range(self.n):
                for l, c in zip(self.labels, self.label_counts):
                    iw = (i + self.idx_counter) % c
                    batch_x[batch_idx, ...] = self.X_dict[l][self.idx_dict[l][iw], ...]
                    batch_x[batch_idx, ...] += torch.randn_like(batch_x[batch_idx, ...]) * self.sigma
                    batch_y[batch_idx, ...] = nn.functional.one_hot(l, num_classes=self.n_classes) if self.to_one_hot else l
                    batch_idx += 1
            
                
            self.idx_counter += self.n
            return batch_x, batch_y
    
        self.idx_counter = 0
        raise StopIteration
        
        
        
    
class LinearTrainer:
    def __init__(self, model) -> None:
        self.model = model
        
    def train(self, X_train, y_train, X_val, y_val, n_epochs=100, lr=1e-3): 
        n_classes = len(torch.unique(y_train))     
        self.n_classes = n_classes   
        self.le = LabelEncoder()
        y_train = torch.from_numpy(self.le.fit_transform(y_train))
        
        y_train_weights = torch.zeros(len(self.le.classes_))
        
        for v in y_train.unique():
            y_train_weights[v] = torch.sum(y_train == v)
            
        print('Train counts: ', y_train_weights.tolist())
        y_train_weights = 1 / y_train_weights.type(torch.float32)
        y_train_weights /= y_train_weights.sum()      
        
        
        y_val = torch.from_numpy(self.le.transform(y_val))
        y_train = nn.functional.one_hot(y_train).type(torch.float32) if n_classes > 2 else y_train.type(torch.float32)
        
        
        X_val = X_val.cuda()
        y_val = nn.functional.one_hot(y_val).type(torch.float32).cuda() if n_classes > 2 else y_val.type(torch.float32).cuda()
        # print(X_train.device, X_test.device, y_train.device, y_test.device)
        generator = torch.Generator(device='cuda')
        self.batch_size = 128
        self.loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, generator=generator, shuffle=True)
        optim = torch.optim.Adam(params=self.model.parameters(), lr = lr)
        loss_fn = nn.CrossEntropyLoss(weight=y_train_weights) if n_classes > 2 else nn.BCELoss()
        
        prev_test_loss = 1e12
        loss_avg = None
        for n in range(n_epochs):            
            self.model.train()
            with torch.set_grad_enabled(True):
                for x_batch, y_batch in self.loader:
                    x_batch = x_batch.cuda()
                    x_batch += torch.randn_like(x_batch)*0.1
                    y_batch = y_batch.cuda()
                    y_pred = self.model(x_batch)
                    if n_classes == 2: 
                        y_pred = y_pred[:, 0]
                        # weight = torch.zeros_like(y_batch)
                        # weight[y_batch == 0] = y_train_weights[0]
                        # weight[y_batch == 1] = y_train_weights[1]
                        # loss_fn = nn.BCELoss(weight=weight)
                        loss = loss_fn(y_pred, y_batch)
                    else:
                        loss = loss_fn(y_pred, y_batch)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
            #print accuracy
            self.model.eval()
            # if n_classes == 2: 
            #     weight = torch.zeros_like(y_val)
            #     weight[y_val == 0] = y_train_weights[0]
            #     weight[y_val == 1] = y_train_weights[1]
            #     loss_fn = nn.BCELoss(weight=weight)
            with torch.set_grad_enabled(False):
                y_pred = self.model(X_val)
                if n_classes == 2: y_pred = y_pred[:, 0]
                test_loss = loss_fn(y_pred, y_val)  
                if n_classes == 2: test_loss = test_loss.sum()
                
                if loss_avg == None: 
                    loss_avg = test_loss
                else:
                    p = 0.8
                    loss_avg = loss_avg * (1-p) + p*test_loss             
                if loss_avg > prev_test_loss: return                
                prev_test_loss = loss_avg
                
                if n_classes > 2:     
                    y_true = torch.argmax(y_val, dim=1)
                    y_pred = torch.argmax(y_pred, dim=1)
                else:
                    y_pred = y_pred > 0.5
                    y_true = y_val  > 0.5  
                
                print(f'Epoch {n} validation accuracy: {torch.sum(y_true == y_pred) / y_true.shape[0] * 100: .2f} (loss={test_loss: .4f}) ')
                
    def test_acc(self, X_test, y_test):
        self.model.eval()
        with torch.set_grad_enabled(False):
            y_test = torch.from_numpy(self.le.transform(y_test))
            y_test = nn.functional.one_hot(y_test.cuda()).type(torch.float32) if self.n_classes > 2 else y_test.cuda().type(torch.float32)
            X_test = X_test.cuda()
            y_pred = self.model(X_test)
            
            auc = metrics.roc_auc_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
            
            y_true = torch.argmax(y_test, dim=1) if self.n_classes > 2 else y_test.cuda()
            y_pred = torch.argmax(y_pred, dim=1) if self.n_classes > 2 else y_pred[:,0].cuda()
            if self.n_classes == 2:
                y_true = y_true > 0.5
                y_pred = y_pred > 0.5
        return torch.sum(y_true == y_pred) / y_true.shape[0], auc
    
    def num_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
        
        
                


EN_LDA_DR = False

results = {}

for d in DATASETS:
    print(
        "---------\n"
        f"{d}\n"
        "---------\n"
    )
    fname = f'data/ws-{d}-mnist3d-Q=[[0.9, 0.9, 0.9]]-T=[48.0, 48.0, 48.0]-DCT=True.pkl'
    with open(fname, 'rb') as file:
        X_train, y_train, X_test, y_test, X_val, y_val = pkl.load(file)
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))
        y_val = torch.from_numpy(y_val.astype(np.float32))
        
    X_train = torch.reshape(X_train, (X_train.shape[0], -1))    
    X_test = torch.reshape(X_test, (X_test.shape[0], -1))  
    X_val = torch.reshape(X_val, (X_val.shape[0], -1))      
    # X_train = torch.mean(X_train, dim=(1,2,3))    
    # X_test = torch.mean(X_test, dim=(1,2,3))  
    # X_val = torch.mean(X_val, dim=(1,2,3))  
    n_classes = len(torch.unique(y_train)) 
    print(f'{n_classes=}') 
    


    # X_train = X_train / torch.max(X_train, dim=1, keepdim=True)[0]
    # X_test = X_test / torch.max(X_test, dim=1, keepdim=True)[0]
    # X_val = X_val / torch.max(X_val, dim=1, keepdim=True)[0]

    mu = torch.mean(X_train, axis=0)
    std = torch.std(X_train, axis=0)

    X_train = (X_train - mu)/std
    X_test = (X_test - mu)/std
    X_val = (X_val - mu)/std
    
        
    # if EN_LDA_DR:
    #     lda = LDA(priors=[1/n_classes for _ in range(n_classes)])
    #     X_train = lda.fit_transform(X_train, y_train)
    #     X_test = lda.transform(X_test)
        
    # clf = SVC(verbose=False, probability=True)

    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # y_prob = clf.predict_proba(X_test) if n_classes > 2 else clf.predict_proba(X_test)[:, 1]


    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, y_pred, digits=3)}\n"
    #     f"AUC={metrics.roc_auc_score(y_test, y_prob, multi_class='ovo')}"
    # )
    
    print(X_train.shape)
    net = DeepClassifier(X_train.shape[1], [4096, 8192, 2048, 1024], n_classes)
    trainer = LinearTrainer(net)
    print(trainer.num_trainable_parameters())
    trainer.train(X_train, y_train, X_val, y_val, n_epochs=100, lr=5e-5)
    acc, auc = trainer.test_acc(X_test, y_test)
    results[d] = {'acc': acc.item(), 'auc': auc, 'n_classes': n_classes}
    print(f'Test Accuracy: {acc: .3f}, AUC: {auc: .3f}')
  
  
import pprint    
pprint.pprint(results)