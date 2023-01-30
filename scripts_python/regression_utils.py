import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from file_utils import saveModelPkl, loadModelPkl

LINEAR = 1
DECISSION_TREE = 2
RANDOM_FOREST = 3

def createRegressor(type: int, adaboost: bool = False):
    """This function creates a regression algorithm to be used later
    If adaboost = True, then it will feed the regressor into an AdaBoostRegressor
    """
    global __regressor__

    if type == LINEAR:
        __regressor__ = LinearRegression()
    elif type == DECISSION_TREE:
        __regressor__ = DecisionTreeRegressor()
    elif type == RANDOM_FOREST:
        __regressor__ = RandomForestRegressor(max_features="sqrt")
    else:
        __regressor__ = None

    if adaboost:
        estimator = __regressor__
        __regressor__ = AdaBoostRegressor(estimator)

def fitRegressor(X_train, y_train):
    """Fits the training data to the regression algorith"""
    if __regressor__ is None:
        raise RuntimeError('Regression algorithm not created')
    
    # Convert the data from a column vector to a 1D array
    y_train = np.ravel(y_train)

    __regressor__.fit(X_train, y_train)

def predict(X_test):
    """Uses the regressor to predict data"""
    if __regressor__ is None:
        raise RuntimeError('Regression algorithm not created')

    return __regressor__.predict(X_test)

def saveRegressor(filename: str = None):
    """Saves the trained model to disk"""
    if __regressor__ is None:
        raise RuntimeError('Regression algorithm not created')

    if filename is None:
        saveModelPkl(__regressor__)
    else:
        saveModelPkl(__regressor__, filename)

def loadRegressor(filename: str = None):
    """Loads the trained regressor from disk"""
    global __regressor__

    if filename is None:
        __regressor__ = loadModelPkl()
    else:
        __regressor__ = loadModelPkl(filename)

    if __regressor__ is None:
        raise RuntimeError('Pickle file not found')

# Show a message if the script is run by itself
if __name__ == '__main__':
    print("This script is not desingned to be standalone.")
