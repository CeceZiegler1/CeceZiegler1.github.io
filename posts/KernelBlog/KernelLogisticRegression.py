import numpy as np 
from scipy.optimize import minimize

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)

class KernelLogisticRegression:

    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs

    def fit(self, X, y):
        X_ = pad(X)
        self.X_train = X_
    
        
        #K = self.kernel(X_, X_, **self.kernel_kwargs)
    
        self.y_train = y
        # Compute kernel matrix
        km = self.kernel(X_, X_, **self.kernel_kwargs)
        n = km.shape[0]
        #selecting random v to start
        v0 = np.random.randn(n)
    
        # Minimize empirical risk using scipy.optimize.minimize()
        res = minimize(self.empirical_risk, v0, args=(km, y))
    
        # Save the optimal weights 
        self.v = res.x
    
    def sigmoid(self, z):# I give credit for this method to the class notes from 2/20
        return 1 / (1 + np.exp(-z))

    def logistic_loss(self, y_hat, y): #I give credit for this method to the class notes from 2/20
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def empirical_risk(self, v, km, y):
        predictions = km @ v
        loss = self.logistic_loss(predictions, y)
        return np.mean(loss)

    def predict(self, X):
        km = self.kernel(pad(X), self.X_train, **self.kernel_kwargs)
        y_hat = np.dot(km, self.v)
        return (y_hat > 0).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)