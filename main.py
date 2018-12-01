import sys
from reader import Reader
from normalizer import Normalizer
from extractor import Extractor
from reducer import Reducer
from regressor import Regressor

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, SGDRegressor, LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import RegressorMixin

if __name__ == '__main__':
    reader = Reader('all_labels.txt', race='C', sex='M')
    normalizer = Normalizer()
    extractor = Extractor() # pasar parametros aca si es necesario
    reducer = Reducer() # pasar parametros aca si es necesario
    alphas = np.logspace(-2, 1, 20)
    gammas = np.logspace(-2, 1, 20)
    print(alphas)
    parametros = {
        # 'alpha': alphas,
        'gamma': gammas,
    }
    # models = GridSearchCV(Ridge(), parametros)
    models = GridSearchCV(SVR(), parametros)
    # models = GridSearchCV(RandomForestRegressor(), parametros)

    X, y = reader.read()
    # normalizer.normalize(X)  # in-place
    features = extractor.extract(X)
    # features = X.reshape((-1, 172))
    print(features)
    x = reducer.reduce(features)
    val = int(0.2 * len(x))
    validation_x = x[ : val]
    validation_y = y[ : val]
    xx = x[val : ]
    yy = y[val : ]
    models.fit(validation_x, validation_y)
    model = models.best_estimator_
    print(model)

    splits = 5
    kf = KFold(n_splits=splits, random_state=None, shuffle=False)
    scores = []
    for train_index, test_index in kf.split(xx):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = xx[train_index], xx[test_index]
        y_train, y_test = yy[train_index], yy[test_index]
        # print(y_test)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # print(y_pred)
        score = model.score(x_test, y_test)
        scores.append(score)
        print('SCORE:', score)
        print('MSE:', mean_squared_error(y_test, y_pred))
        print('MAE:', mean_absolute_error(y_test, y_pred))
        print('')
    scores = np.array(scores)
    print('\nSTD:', scores.std())
    print('MEAN:', scores.mean())
