from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


data=pd.read_csv('fashionmist/fashion-mnist_train.csv')
Y=data.iloc[:,0].values
X=data.iloc[:,1:].values


oh=OneHotEncoder()
Y=oh.fit_transform(Y)
Xtrain=X.T
Ytrain=Y.T


inplayer=Xtrain.shape[0] ## 784
h1layer=256
h2layer=32
labels=Ytrain.shape[0] ## 10
W=np.random.normal(0.0,1.3,size=(inplayer,h1layer))
W1=np.random.normal(0.0,2.3,size=(h1layer,h2layer))
W2=np.random.normal(0.0,1.9,size=(h2layer,labels))

lr=0.01
batch=256
epochs=5


for e in range(epochs):
    tot_loss=0
    i=0
    while(i<60000):
        start=i
        end=i+batch
        X,Y=Xtrain[:,start:end],Ytrain[:,start:end]
        
        ## forward pass
        H1=1/(1+np.exp(-np.dot(W.T,X)))
        H2=1/(1+np.exp(-np.dot(W1.T,H1)))
        exp=np.exp(np.dot(W2.T,H2))
        Yhat=exp/(np.sum(exp,axis=0))
        loss=-np.mean(np.sum(Y*np.log(Yhat),axis=0))
        tot_loss+=loss
        
        ## required functions to find gradients followed by finding the gradients 
        err=Y-Yhat
        dW2=np.dot(H2,err.T)
        res2= H2*(1-H2)*(np.dot(W2,err))
        dW1=np.dot(H1,res2.T)
        res1=H1*(1-H1)*(np.dot(W1,res2))
        dW=np.dot(X,res1.T)
        
        ### updating the gradients 
        W2=-(lr/batch)*(dW2)
        W1=-(lr/batch)*(dW1)
        W=-(lr/batch)*(dW)
        
        i=end
    print('epoch{}--loss= {}'.format(e,tot_loss))
     
        
        
        
        
        
        
        
        