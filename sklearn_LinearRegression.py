# use python to write up a Linear Regression model using sklearn package
# use Boston house price as the datasets to be analyzed

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 1. load in the boston house price data
boston = datasets.load_boston()
# examine the data
# boston dataset has 2 parts:
# 1) data: all explanatory vairables
# 2) target: response - house price
print(boston.data.shape)   # (506, 13) 506 observations, 13 attributes
print(boston.target.shape) # (506, )
print(boston.feature_names)

# 2. get rid of unreasonable data
plt.scatter(x = boston.data[:, 5], y = boston.target)
plt.show()
# simply taking the fifth col (NOX) as explanatory variable, plot scatterplot against house price
# we can see outliers (house price >= 50)
# get rid of those
mdata = boston.data
mtarget = boston.target
# boolean indexing
mdata = mdata[mtarget < 50]
mtarget = mtarget[mtarget < 50]
# plot again
plt.scatter(x = mdata[:, 5], y = mtarget)
plt.show()

# 3. split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(mdata, mtarget, random_state = 123)
# test the size
print(f'X_train size :{X_train.shape}, X_test size :{X_test.shape}, y_train size :{y_train.shape}, y_test size :{y_test.shape}')

# 4. fit the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. once fit, print out parameters of the fitted model
print(f'Intercept:{model.intercept_}, Coefficients:{model.coef_}')

# 6. predict with X_test dataset
predict_y = model.predict(X_test)
print(f'Predicted y:{predict_y}')

# 7. evaluate
print(f'Score: {r2_score(y_test, predict_y)}')
# when R^2 < 1, the bigger the better.
# now we get R^2 = 0.79, let's see whether normalizing can make it better

# 8. normalize training X, and test X dataset
norm = StandardScaler()
norm.fit(X_train, y_train)
# now that it's been fitted, we normalize the data
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

# 9. fit MLR to the normalized data
model_norm = LinearRegression()
model_norm.fit(X_train_norm, y_train)
predict_y_norm = model_norm.predict(X_test_norm)
print(f'Normalized Score: {r2_score(y_test, predict_y_norm)}')
# 2 scores have no difference, no need to normalize

# 10. sort coefficients
# return list of index that would sort an array
# fancy indexing
sort = boston.feature_names[np.argsort(model.coef_)]
print (sort)
print (boston.DESCR)