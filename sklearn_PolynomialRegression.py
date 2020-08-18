# Polynomial Regression overview
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# mock data
x = np.random.uniform(-3, 3, 100)
# convert to col vector, now treated as one explanatory variable
X = x.reshape(-1, 1)
# a Quadratic formula plus a normal distribution noise
y = 0.5* X **2 + X + 2 + np.random.normal(0, 1, 100).reshape(-1, 1)
# plot scatter
plt.scatter(X, y)
plt.show()


# now first try fitting a linear regression
linreg = LinearRegression()
linreg.fit(X, y)
predict = linreg.predict(X)    # here we are not splitting data into training & test
print(f'linearregression score : {linreg.score(X, y)}')
print(f'linearregression intercept : {linreg.intercept_}, linearregression coefficient: {linreg.coef_}')

# plot again
plt.plot(X, predict, color = 'r')
plt.scatter(X, y)
plt.title('Linear regression')
plt.show()


# so try apply a polynomial regression
# use Quadratic formula (y = ax^2 + bx + c)
# instead of (y = ax + b)
# for column vector X, we take square of every element in X, and combine original X and X^2
X2 = np.hstack([X, X**2])    # vectorized calculation
print (f'originalXshape:{X.shape}, updatedXshape:{X2.shape}')

# now treat it as a linear regression
poly = LinearRegression()
poly.fit(X2, y)
predict2 = poly.predict(X2)
print(f'Polynomialregression score : {poly.score(X2, y)}')
print(f'Polynomialregression intercept : {poly.intercept_}, Polynomialregression coefficient: {poly.coef_}')

# plot again
# scatter can just use original data
# tricky part is to sort x! or else won't be a curve
plt.scatter(x, y)
plt.plot(np.sort(x), predict2[np.argsort(x)], color = 'r')
plt.title('Polynomial regression')
plt.show()


# scikit-learn polynomial regression
import numpy as np
import matplotlib.pyplot as plt
# first create poly model
poly = PolynomialFeatures(degree = 2)
# fit the model to the training X
poly.fit(X)
# transform original X to the updated polynomial X
polyX = poly.transform(X)


print(X[:5, :])
print(polyX[:5, :])

# this is original X
# [[-1.71123902]
#  [-1.72788391]
#  [-0.91883326]


# this is poly X
# [[ 1.00000000e+00 -1.71123902e+00  2.92833897e+00]
#  [ 1.00000000e+00 -1.72788391e+00  2.98558279e+00]
#  [ 1.00000000e+00 -9.18833256e-01  8.44254552e-01]
# 1st col: X**0 (X to the power of 0)
# 2nd col: X
# 3rd col: X**2

# once transformed, fit to linear regression again
sklpoly = LinearRegression()
sklpoly.fit(polyX, y)
predict3 = sklpoly.predict(polyX)
print(f'sklearn Polynomialregression score : {sklpoly.score(polyX, y)}')
print(f'sklearn Polynomialregression intercept : {sklpoly.intercept_}, sklearn Polynomial coefficient: {sklpoly.coef_}')

# plot
# note when plotting scatter, use the original x
# don't use the updated x matrix
plt.scatter(x, y)
# note remember to reorder when plotting line plot, cuz linking them is based on the given order
plt.plot(x, predict3[np.argsort(x)], color = 'r')
plt.title('sklearn Polynomial regression')

# sklearn polynomialfeatures gives same result as our manual written one

