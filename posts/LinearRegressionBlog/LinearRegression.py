import numpy as np

class LinearRegression():

    def pad(self, X): #I give credit for this method to the help section of the assignment page
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def predict(self, X, w):
        X_ = self.pad(X)
        return X_@w

    def fit_analytic(self, X, y):
        X_ = self.pad(X)
        w = np.random.random((X_.shape[1]-1))#initializing weight vector
        self.w = w

        history = []
        self.history = history#history of score/accuracy

        loss = np.inf#initializing loss
        self.loss = loss

        self.w = np.linalg.inv(X_.T@X_)@X_.T@y #updating w

        y_pred = self.predict(X, self.w)

        self.loss = np.mean((y-y_pred)**2)

        self.history.append(self.score(X, y))

    def fit_gradient(self, X, y, alpha, max_epochs):
        X_ = self.pad(X)
        w = np.random.random(X_.shape[1])#initializing weight vector
        self.w = w

        p = (X_.T).dot(X_)
        self.p = p

        q = (X_.T).dot(y)
        self.q = q

        loss = np.inf#initializing loss
        self.loss = loss


        score_history = []
        self.score_history = score_history#history of score/accuracy


        for i in range(max_epochs):
            
            #computing gradient from gradient function below
            #grad = (self.p@self.w) - self.q
            grad = self.gradient(self.p, self.q, self.w)
            self.grad = grad

            #Updating weight
            self.w = self.w - (alpha * self.grad) #moving down the gradient, thus subtraction

            #print(np.linalg.norm(grad))  
            #updating history by calling score method
            self.score_history.append(self.score(X, y))

            y_pred = self.predict(X, self.w)
                
            self.loss = np.mean((y-y_pred)**2)

    def score(self, X, y):
        y_hat = self.predict(X,self.w)
        return 1- ((np.sum((y_hat-y)**2))/(np.sum((y-y.mean())**2)))

    def gradient(self, p, q, w):
        gradient = (p@w - q)
        return gradient