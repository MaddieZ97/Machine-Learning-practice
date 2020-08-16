# manually write up Simple Linear Regression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math

# Simple linear regression
class SimLinear():
    '''
    fit a simple linear regression model
    '''
    def __init__(self):
        '''
        intialize a simple linear regression model
        '''
        # coefficient
        self._a = None
        # intercept
        self._b = None

    def fit(self, x_train, y_train):
        '''
        fit model to training x and training y
        :param x_train: training dataset x (ndarray)
        :param y_train: training dataset y (ndarray)
        :return: fitted model, model itself
        '''
        # calculate mean
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # calculate parameter a and b (formula given by Least Square Error)
        numerator = np.sum((x_train - x_mean).dot(y_train - y_mean))   # don't need to be dot product here, since it's scalar
        denominator = np.sum((x_train - x_mean).dot(x_train - x_mean))
        self._a = numerator/denominator
        self._b = y_mean - self._a * x_mean

    def predict(self, x_test):
        '''
        predict y based on test data x
        :param x_test: test data x (ndarray)
        :return: predicted y
        '''
        # y = ax + b
        return self._a * x_test + self._b


    # all methods start with (self, ....)

    def mse(self, predict_y, y_test):
        '''
        mean squared error
        :param y_predict: predicted y
        :param y_test: test data y
        :return: mean squared error
        '''
        return (np.sum((y_test - predict_y) ** 2)) / len(y_test)

    def rmse(self, predict_y, y_test):
        '''
        rooted mean squared error
        :param y_test:
        :return:
        '''
        return np.sqrt((np.sum((y_test - predict_y) ** 2)) / len(y_test))

    def mae(self, predict_y, y_test):
        '''
        mean absolute error
        :param y_test:
        :return:
        '''
        return (np.sum(np.absolute(y_test - predict_y))) / len(y_test)

    def rsquare(self, predict_y, y_test):
        '''
        rsquared score
        :param predict_y: predicted y
        :param y_test: test data y
        :return: rsquare score
        '''
        r2 = 1- self.mse(predict_y, y_test) / np.var(y_test)
        return r2

# load data for SLR
boston = datasets.load_boston()

# get rid of unreasonable data
mdata = boston.data
mtarget = boston.target
# boolean indexing
mdata = mdata[mtarget < 50]
mtarget = mtarget[mtarget < 50]
# get the fifth col (for SLR)
mdata = mdata[:, 5]


#split data
X_train, X_test, y_train, y_test = train_test_split(mdata, mtarget, random_state = 123)

# fit model
SLR = SimLinear()
SLR.fit(X_train, y_train)
predict_y = SLR.predict(X_test)

a = SLR.mse(predict_y, y_test)
b = SLR.rmse(predict_y, y_test)
c = SLR.mae(predict_y, y_test)
d = SLR.rsquare(predict_y, y_test)
print(f'Mean squared error: {a}')
print(f'Rooted mean squared error: {b}')
print(f'Mean absolute error: {c}')
print(f'rsquared score: {d}')
# r2 score < 1, the larger the better



# Multiple Linear regression
class MultipleLinear():
    '''
    fit a multiple linear regression model
    '''
    def __init__(self):
        self.coef_ = None
        self.inter_ = None
        self.theta_ = None

    def fit(self, x_train, y_train):
        '''
        fit a multiple linear regression model
        :param x_train:
        :param y_train:
        :return:
        '''
        # create the x_b matrix, adding the all-1 column for intercept theta
        ones = np.ones((len(x_train), 1))   # double brackets to state size
        x_b = np.hstack([ones, x_train])    # [...] to combine
        self.theta_ = (np.linalg.inv((x_b.T).dot(x_b))).dot(x_b.T).dot(y_train)
        self.inter_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        return self

    def predict(self, x_test):
        '''
                function: return predicted value y
                :param x_test: test value, value to predict
                :return: predicted value
                '''
        # again add the all-1 column to the xvalue for prediction
        ones = np.ones((len(x_test), 1))  # double brackets
        test_b = np.hstack([ones, x_test])
        # return predicted
        y_pred = test_b.dot(self.theta_)  # order does not matter
        return y_pred

    def score(self, x_test, y_test):
        '''
        function: to test how well our model fits
        :param x_test: test data
        :param y_test: true response for test data
        '''
        y_pred = self.predict(x_test)
        # r_2
        mse = (np.sum((y_test - y_pred) ** 2)) / len(y_test)
        # by simplified r_squared formula
        r_2 = 1 - mse / np.var(y_test)
        return r_2




# load data for MLR
boston = datasets.load_boston()
# get rid of unreasonable data
data = boston.data
target = boston.target
# boolean indexing
data = data[target < 50]
target = target[target < 50]


#split data
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state = 123)


# fit model
MLR = MultipleLinear()
MLR.fit(X_train, y_train)
predict_y = MLR.predict(X_test)
d = MLR.score(X_test, y_test)
print(f'rsquared score: {d}')
# r2 score < 1, the larger the better




