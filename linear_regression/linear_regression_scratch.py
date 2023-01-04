# Importing libraries
  
from lr import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
  
import matplotlib.pyplot as plt
  
# Linear Regression
  
  
# driver code
  
def main() :
      
    # Importing dataset
      
    df = pd.read_csv( "salary_data.csv" )
  
    X = df.iloc[:,:-1].values
  
    Y = df.iloc[:,1].values
      
    # Splitting dataset into train and test set
  
    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/3, random_state = 0 )
      
    # Model training
      
    model = LinearRegression( iterations = 1000, learning_rate = 0.01 )
  
    model.fit( X_train, Y_train )
      
    # Prediction on test set
  
    Y_pred = model.predict( X_test )
      
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) ) 
      
    print( "Real values      ", Y_test[:3] )
      
    print( "Trained W        ", round( model.W[0], 2 ) )
      
    print( "Trained b        ", round( model.b, 2 ) )
      
    # Visualization on test set 
      
    plt.scatter( X_test, Y_test, color = 'blue' )
      
    plt.plot( X_test, Y_pred, color = 'orange' )
      
    plt.title( 'Salary vs Experience' )
      
    plt.xlabel( 'Years of Experience' )
      
    plt.ylabel( 'Salary' )
      
    plt.show()
     
if __name__ == "__main__" : 
      
    main()