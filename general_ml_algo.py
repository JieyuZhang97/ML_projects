import numpy as np 

class generalMlAlgo():
    def __init__(self, learning_rate, iterations, convergence_tolerance=1e-10, stop_update_weights=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.convergence_tolerance = convergence_tolerance
        self.converged = False
        self.stop_update_weights = stop_update_weights
    # Function for model training

    def fit(self, X, Y):
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        # gradient descent learning
        # TODO: either use iternations as an indicator or stops when reaches global minimum
        for i in range(self.iterations):
            self.update_weights()
            if self.__potential_convergence:
                break

        return self

    # Helper function to update weights in gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X)
        # use gradient descent 
        # calculate gradients
        # this part is different from the notes of Andrew Ng's. The function he uses has an extra 2 in the denominator
        # the dot: [x1,x2,x3,...,xk]T * [y1,y2,y3,...yk]
        dW = - (2*(self.X.T).dot(self.Y-Y_pred))/self.m
        db = - 2*np.sum(self.Y-Y_pred)/self.m
        # TODO: determine if global minimum has reached 
        if np.allclose(self.W, dW, rtol=self.convergence_tolerance) and np.allclose(self.b, db, rtol=self.convergence_tolerance):
            self.__potential_convergence += 1
        else:
            self.__potential_convergence = 0
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        if self.__potential_convergence >= self.stop_update_weights:
            self.converged = True

    # Hypothetical function  h(x)
    def predict(self, X):
        # linear regression
        return X.dot(self.W) + self.b
