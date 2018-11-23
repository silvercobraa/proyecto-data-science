from reader import Reader
from normalizer import Normalizer
from extractor import Extractor
from reducer import Reducer
from regressor import Regressor

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    reader = Reader('train_test_files/All_labels.txt', race='A', sex='F')
    normalizer = Normalizer()
    extractor = Extractor() # pasar parametros aca si es necesario
    reducer = Reducer() # pasar parametros aca si es necesario
    model = Regressor('ridge') # pasar parametros aca si es necesario

    X, y = reader.read()
    normalizer.normalize(X)  # in-place
    features = extractor.extract(X)
    x = reducer.reduce(features)

    splits = 5
    kf = KFold(n_splits=splits, random_state=None, shuffle=False)
    scores = []
    for train_index, test_index in kf.split(x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
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
