import numpy as np

class LinearRegression():

    def pad(self, X): #I give credit for this method to the help section of the assignment page
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def predict(self, X, w): # I give credit for this method to the class notes from 2/20
        #input: feature matrix, X, wegiht vector, w
        #output: prediction vector

        #this method computes the predicted value, y_hat by computing the dot product of X and w.
        X_ = self.pad(X)
        return X_@w

    def fit_analytic(self, X, y):
        #Input: Feature matrix, X and target vector, y
        #Output: This method has no return type

        #This method computes the value of the weight vector, w, using the analytical formula
        #we calculated in class. It then computes the score and keeps track of the score history
        #with an array

        X_ = self.pad(X)
        w = np.random.random((X_.shape[1]-1))#initializing weight vector
        self.w = w

        history = []
        self.history = history#history of score/accuracy

        loss = np.inf#initializing loss
        self.loss = loss

        self.w = np.linalg.inv(X_.T@X_)@X_.T@y #updating w using analytical formula, code from clas notes

        y_pred = self.predict(X, self.w)

        self.loss = np.mean((y-y_pred)**2)

        self.history.append(self.score(X, y))#updating loss history array

    def fit_gradient(self, X, y, alpha, max_epochs):
        #Input: Feature matrix, X, target vector, y, learning rate, alpha, number of iterations, max_epochs
        #Output: this method has no return type

        #This method converges by using gradient decent to calculate the value of the weight vector
        #It calculates values of p and q once in the method to make computing the gradient
        #more efficient at each iteration. It then computes the score and keeps track of the 
        #score over time in an array

        X_ = self.pad(X)
        w = np.random.random(X_.shape[1])#initializing weight vector
        self.w = w

        p = (X_.T).dot(X_)#computing the p value
        self.p = p

        q = (X_.T).dot(y)#computing the q value
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

            #updating history by calling score method
            self.score_history.append(self.score(X, y))

            y_pred = self.predict(X, self.w)
                
            self.loss = np.mean((y-y_pred)**2)

    def score(self, X, y):
        #input: this method takes in the feature matrix X and target vector y
        #output: this method returns a float that is the value of the score

        #the method computes the score using the formula we calculated in our notes in class

        y_hat = self.predict(X,self.w)
        return 1- ((np.sum((y_hat-y)**2))/(np.sum((y-y.mean())**2)))

    def gradient(self, p, q, w):
        #input: matrix, p, vector, q, vector, w
        #output: vector, gradient

        #this method computes the gradient using the gradient formula we calculated in class
        #It saves time, by using the pre calculated values of p and q

        gradient = (p@w - q)
        return gradient