from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# load data for MLR
boston = datasets.load_boston()
# get rid of unreasonable data
data = boston.data
target = boston.target
# boolean indexing
data = data[target < 50]
target = target[target < 50]
# split data
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=123)



# create normalized object
sca = StandardScaler()
# fit model
sca.fit(X_train, y_train)

# now model is already fitted, we normalize the data
X_train_norm = sca.transform(X_train)
X_test_norm = sca.transform(X_test)



# now use a new method: KN regression
kng = KNeighborsRegressor()
kng.fit(X_train_norm, y_train)
ks = kng.score(X_test_norm, y_test)      #note: parameters diff from r2_score (y_test, pred)

print(f'KNeighborsRegressor score: {ks}')  # 0.7885

# create GridSearchCV instance
# use GridSearch to determine the most optimized parameters
# GridSearch is tied to KNN

param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]


# only apply to KNeighborsRegressor case which has hyperparameters
xt = KNeighborsRegressor()
gs = GridSearchCV(xt, param_grid,
                  n_jobs = -1,
                  verbose = 1)

# train gridsearch model
# X_train is the normalized training set
gs.fit(X_train_norm, y_train)

# after training, we get the best optimized hyperparameter
print (f'best hyperparameter: {gs.best_params_}')

# obtain linear regression model with the best hyperparameter
best_lr = gs.best_estimator_

# now get score
sc = best_lr.score(X_test_norm, y_test)
print(f'KNeighborsRegressors with best hyperparameters score:{sc}')
