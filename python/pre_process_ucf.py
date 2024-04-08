from sepws.dataprocessing.ucf import save_train_test, read_train_test

# save_train_test()

X_train, y_train, X_test, y_test = read_train_test()

Ns = []

for x in X_train: 
    print(x.shape)
    Ns.append(x.shape[-1])
for x in X_test:
    print(x.shape)
    Ns.append(x.shape[-1])
    
    
import numpy as np
print(np.max(Ns), np.min(Ns))