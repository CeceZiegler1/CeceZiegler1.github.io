import numpy as np

#https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
#The link above is where I got the code for many of my mathematical functions

class LogisticRegression():

    def pad(self, X): #I give credit for this method to the help section of the assignment page
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def predict(self, X, w): # I give credit for this method to the class notes from 2/20
        y_hat = X@w
        return y_hat
    
    def sigmoid(self, z):# I give credit for this method to the class notes from 2/20
        return 1 / (1 + np.exp(-z))
    
    def logistic_loss(self, y_hat, y): #I give credit for this method to the class notes from 2/20
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def empirical_risk(self, X, y, log_loss, w): # I give credit for this method to the class notes from 2/20
        y_hat = self.predict(X, w)
        return self.logistic_loss(y_hat, y).mean() #returning the total loss/empirical risk

    def fit(self, X, y, alpha, max_epochs):
        #This method takes in a feature matrix, target vector, learning rate and max
        #number of epochs as parameters and returns nothing. In this method the weight
        #vector, w will continue to be updated until the loss is so close to the previous
        #loss updates are no longer making a difference. The weight vector is updated
        #by computing the gradient, and aspects such as the loss history and accuracy
        #are tracked throughout the method but not returned
        X = self.pad(X)
        w = np.random.random(X.shape[1])#initializing weight vector
        self.w = w

        loss = np.inf#initializing loss
        self.loss = loss

        history = []
        self.history = history#history of score/accuracy

        loss_history = []
        self.loss_history = loss_history#used to track loss history

        done = False
        prev_loss = np.inf
        self.prev_loss = prev_loss#helps determine when to terminate

        while not done:

            for i in range(max_epochs):
                z = self.predict(X, self.w)
                y_predictor = self.sigmoid(z)#creating y_hat predictor

                #Computing gradient using derivative
                #computing gradient from gradient function below
                grad = self.gradient(self.w, X, y)

                #Updating weight
                self.w = self.w - (alpha * grad) #moving down the gradient, thus subtraction
                
                #updating history by calling score method
                self.history.append(self.score(X, y))
                
                #computing logistic loss to calculate total empirical loss
                log_loss = self.logistic_loss(y_predictor, y)
                self.loss = self.empirical_risk(X, y, log_loss, self.w)#This is the total loss to track 
                self.loss_history.append(self.loss)

                if np.isclose(self.loss, self.prev_loss):# checking if loss is close enough to previous to terminate
                    done = True
                else:
                    self.prev_loss = self.loss

                
    
    def score(self, X, y):
        #This method takes in the feature matrix X and target vector Y and returns
        #a double that is the average score/ accuracy of correct predictions
        y_hat = ((np.dot(X, self.w)) > 0) *1.
        return (y_hat == y).mean()

    
    def fit_stochastic(self, X, y, alpha, max_epochs, momentum, batch_size):
        #This method takes in a feature matrix, target vector, learning rate, max epochs
        #momentum boolean and batch size as parameters and returns nothing. This mehtod
        #will contiuously update the weight vectors, w, until it converges to a minimum
        #by computing the gradient of each batch. If momentum is true, it will at 
        #that to the computation when updating the weights. Throughout the method,
        #loss will be computed and tracked

        X = self.pad(X)
        w = np.random.random(X.shape[1])#creating weight vector
        self.w = w

        loss_history = []
        self.loss_history = loss_history#used to store loss over time

        done = False
        prev_loss = np.inf
        self.prev_loss = prev_loss#helps track when model has convereged


        n = X.shape[0]

        w_history = [[0,0,0],self.w]
        self.w_history = w_history#used to compute the momentum

        B = 0.0#beta is zero unless momentum is true
        i = 2

        if(momentum):
            B = 0.8

        while not done:
            for j in np.arange(max_epochs):
            
                order = np.arange(n)
                np.random.shuffle(order)#getting a random batch

                for batch in np.array_split(order, n // batch_size + 1):
                    x_batch = X[batch,:]#feature matrix based on batch size
                    y_batch = y[batch]#target vector based on batch size
                    grad = self.gradient(self.w, x_batch, y_batch) #computing gradient from gradient function
                    self.w = self.w - (alpha * grad) + (B*(self.w - self.w_history[i-2]))#updating weights with option of momentum
                    self.w_history.append(self.w) 
                    i += 1

                log_loss = self.logistic_loss(self.sigmoid(X@self.w), y)# computing loss with each epoch
                self.loss = self.empirical_risk(X, y, log_loss, self.w)
                self.loss_history.append(self.loss)

                if np.isclose(self.loss, self.prev_loss):#helping to track convergence
                    done = True
                else:
                    self.prev_loss = self.loss

    def gradient(self,w, X, y ):
        #This method takes in a weight vector, feature matrix, and target vector as 
        #parameters and returns the gradient
        gradient = np.mean(((self.sigmoid(X@w)[:,np.newaxis]) - y[:,np.newaxis])*X, axis = 0)
        return gradient

        