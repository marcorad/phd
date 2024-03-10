import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle as pkl
from sklearn import metrics
import torch

fname = 'data/ws-organmnist3d-Q=[[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]]-T=[32.0, 32.0, 32.0].pkl'
with open(fname, 'rb') as file:
    X_train, y_train, X_test, y_test = pkl.load(file)
    
X_train = np.reshape(X_train, (X_train.shape[0], -1))   
X_test = np.reshape(X_test, (X_test.shape[0], -1))    

print(X_train.shape)
print(X_train.dtype)

mu = torch.mean(X_train, axis=0)
std = torch.std(X_train, axis=0)

X_train = (X_train - mu)/std
X_test = (X_test - mu)/std
    
    
    
# feat_red = LDA(priors=[1/11 for _ in range(11)])
clf = SVC(verbose=True, probability=False)

# X_train = feat_red.fit_transform(X_train, y_train)
# X_test = feat_red.transform(X_test)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# y_prob = clf.predict_proba(X_test)


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, y_pred, digits=3)}\n"
    # f"AUC={metrics.roc_auc_score(y_test, y_prob, multi_class='ovo')}"
)