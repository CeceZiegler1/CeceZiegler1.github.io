import numpy as np

#X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

class Perceptron():
    
    
    def fit(self, X, y, max_steps):
        #This method takes in a feature matrix, X, a target vector, y, and the max
        #number of steps for which the algorithm will iterate if it doesn't
        #converge to zero. The method performs matrix multiplications using the dot
        #product after selecting a random weight, w. It then computes the sign of
        # the predictor value and updates the weights accordingly using equation 1
        #The function updates the loss and prints it out, but has no return value
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        w = np.random.random(X_.shape[1])#Want the p from the Xnbyp matrix +1 to the p so that it is w tild/ w-b save it the instance variable
        self.w = w
        loss = 1
        history = []
        self.history = history
        #while loss != 0:
        for i in range(max_steps):
            i = i % X_.shape[0]
            dp = np.dot(w, X_[i,:]) #computing dot product
            y_predictor = np.sign(dp) #computing sign of the predictor value
            self.w = self.w + 1*(y_predictor < 0)*y[i]*X_[i,:] #updating weights
            self.history.append(self.score(X_, y))
            loss = 1-self.score(X_, y)
        print(loss)
            
                
    def score(self, X, y):
        #This method takes as input a feature matrix, X, and a target vector, y.
        # The method computes the dot product of X and weight vector, w, then
        #multiplies it by y and compares it to zero, and computese the mean to
        #get the average accuracy of the weight vector. It returns a number as the average
        return((np.dot(X, self.w)*y) > 0).mean()
    
    def predict(self, X):
        #This method takes in a feature matrix, X. It computes the dot product of
        #the matrix X and weight vector, w. The method returns true in the form of 1
        #if the dot product is greater than 0 and false in the form of 0 otherwise
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        return((np.dot(X_, self.w)) > 0)
    