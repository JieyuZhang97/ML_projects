
import numpy as np
  

class LinearRegression() :
    def __init__( self, learning_rate, iterations ) :
        self.learning_rate = learning_rate
        self.iterations = iterations
    # Function for model training
    def fit( self, X, Y ) :
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros( self.n )
        self.b = 0
          
        self.X = X
          
        self.Y = Y
        # gradient descent learning
        # TODO: either use iternations as an indicator or stops when reaches global minimum
        for i in range( self.iterations ):
            self.update_weights()
              
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
             
        Y_pred = self.predict( self.X )
          
        # calculate gradients  
      
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred )  ) / self.m
       
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
          
        # update weights
      
        self.W = self.W - self.learning_rate * dW
      
        self.b = self.b - self.learning_rate * db
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :
        return X.dot( self.W ) + self.b

    def get_slope(self, X, Y):
        self.W = np.cov(X,Y)/np.var(X)
        return self.W

    def get_intercep(self,X,Y):
        self.b = np.mean(Y) - np.mean(X)*self.W
