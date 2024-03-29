import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from file_utils import saveModelPkl, loadModelPkl

LINEAR = 0
DECISSION_TREE = 1
RANDOM_FOREST = 2
K_NEAREST = 3

def createRegressor(type: int, adaboost: bool = False, n_neighbors: int = 5):
    """This function creates a regression algorithm to be used later
    If adaboost = True, then it will feed the regressor into an AdaBoostRegressor
    """
    global __regressor__

    if type == LINEAR:
        # n_jobs = -1 : use all available processors
        __regressor__ = LinearRegression(n_jobs=-1)
    elif type == DECISSION_TREE:
        __regressor__ = DecisionTreeRegressor()
    elif type == RANDOM_FOREST:
        # n_jobs = -1 : use all available processors
        __regressor__ = RandomForestRegressor(max_features="sqrt", n_jobs=-1)
    elif type == K_NEAREST:
        __regressor__ = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    else:
        __regressor__ = None

    if adaboost:
        if type == DECISSION_TREE:
            # making some modifications to the DTR if used with AdaBoost technique
            __regressor__ = DecisionTreeRegressor(max_depth=12, max_features="sqrt")

        estimator = __regressor__
        __regressor__ = AdaBoostRegressor(estimator, n_estimators=100, learning_rate=0.1)

def fitRegressor(X_train, y_train):
    """Fits the training data to the regression algorithm"""
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
