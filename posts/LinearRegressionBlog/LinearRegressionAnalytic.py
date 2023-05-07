import numpy as np

class LinearRegressionAnalytic:
    def __init__(self):
        self.w = None
        
    def fit(self, X, y):
        X_ = self.pad(X)

        w = np.random.random((X_.shape[1]-1))#initializing weight vector
        self.w = w

        self.w = np.linalg.inv(X_.T@X_)@X_.T@y
        
    def pad(self, X): #I give credit for this method to the help section of the assignment page
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def predict(self, X, w): # I give credit for this method to the class notes from 2/20

        '''takes in a feature matrix X, weight vector w, returns the the dot product of the two'''

        X_ = self.pad(X)
        return X_@w
        
    def score(self, X, y):
        y_hat = self.predict(X,self.w)
        mse = np.mean((y - y_hat)**2)
        return 1 - mse/np.var(y)
        #return 1- ((np.sum((y_hat-y)**2))/(np.sum((y-y.mean())**2)))

# define the RFE function
def rfe(X, y, k):

    if X.shape[1] < k:
        raise ValueError("Not enough features to select.")

    lr = LinearRegressionAnalytic()
    # initialize the set of selected features to be empty
    selected = set()
    # initialize the set of remaining features to be all the features
    remaining = set(range(X.shape[1]))
    # iterate k times
    for i in range(k):
        # compute the optimal weights for each remaining feature
        weights = {}
        for j in remaining:
            features = list(selected) + [j]
            X_subset = X.iloc[:, features]
            lr.fit(X_subset, y)
            weights[j] = lr.w
        # select the feature with the smallest absolute weight
        best_feature = min(weights, key=lambda x: abs(weights[x][-1]))
        # add the best feature to the set of selected features
        selected.add(best_feature)
        # remove the best feature from the set of remaining features
        remaining.remove(best_feature)
    # return the indices of the selected features
    return list(selected)


