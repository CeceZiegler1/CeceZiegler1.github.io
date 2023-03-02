import numpy as np

class LogisticRegression():

    def pad(self, X): #I give credit for this method to the help section of the assignment page
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def predict(self, X, w): # I give credit for this method to the class notes from 2/20
        return X@w
    
    def sigmoid(self, z):# I give credit for this method to the class notes from 2/20
        return 1 / (1 + np.exp(-z))
    
    def logistic_loss(self, y_hat, y): #I give credit for this method to the class notes from 2/20
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def loss(self, X, y, log_loss, w): # I give credit for this method to the class notes from 2/20
        y_hat = self.predict(X, w)
        return self.logistic_loss(y_hat, y).mean() #returning the total loss/empirical risk

    def fit(self, X, y, alpha, max_epochs):
        X = self.pad(X)
        w = np.random.random(X.shape[1])
        self.w = w

        loss = 1
        self.loss = loss

        for i in range(max_epochs):
            z = self.predict(X, self.w)
            y_predictor = self.sigmoid(z)

            #Computing gradient using derivative
            dw = (1/(X.shape[1])) * np.dot(X.T, (y_predictor -y))

            #Updating weight
            self.w = alpha * dw

            log_loss = self.logistic_loss(y_predictor, y)
            self.loss = 1 - loss(X, y, log_loss, w)

        