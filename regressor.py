from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, SGDRegressor, LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import RegressorMixin
import numpy as np

class Regressor(object):
    """Regresor genérico. """
    _models = {
        'ridge': Ridge,
        'svm': SVR,
        'lasso': Lasso,
        'knn': KNeighborsRegressor,
        # etc
    }
    def __init__(self, model, **params):
        self.model = Regressor._models[model](**params)
        # self.model.set_params(params)

    def fit(self, X, y, sample_weight=None):
        return self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight=sample_weight)

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    r = Regressor('svm', gamma='scale')
    print(r.fit(x, y))
    print(r.predict(x))
    print(r.score(x, y))
