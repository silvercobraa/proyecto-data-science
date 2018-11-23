from reader import Reader
import numpy as np
import numpy.linalg as la
from offset import Offset
import pylab as pl

class Normalizer(object):
    '''Clase encargada de normalizar las coordenadas de un dataset, esto es trasladar, rotar y escalar los rostros para llevarlos a un espacio común.'''
    def __init__(self):
        pass

    def _normalize(self, x):
        '''Recibe un vector de 86x2 y lo normaliza (in place).'''
        # la pupila derecha de cada rostro quedará en el origen
        translation = - x[Offset.RIGHT_PUPIL].T
        left_pupil = x[Offset.LEFT_PUPIL].T + translation
        # como la otra pupila esta en el origen, calcular el angulo entre ellas se simplica bastante. Mas aun, no es necesario calcular el angulo directamente, pues podemos sacar los cosenos y senos necesarios para la matriz de rotacion simplemente con las proyecciones en cada eje (o sea, las componentes x e y)
        pupil_distance = la.norm(left_pupil)
        cos0 = left_pupil[0] / pupil_distance
        sen0 = left_pupil[1] / pupil_distance
        # matriz de rotacion para dejar las pupilas paralelas al eje x. Notese que el seno negativo va abajo, ya que queremos rotar en sentido contrario al angulo de elevacion.
        rotation = np.array([[cos0, sen0], [-sen0, cos0]])
        # roto la pupila para poder obtener el factor de escaldo correcto
        scaling = 1 / pupil_distance
        for i in range(len(x)):
            x_i = x[i]
            x[i] = (scaling * (rotation @ (x[i] + translation))).T
        return x

    def normalize(self, X):
        '''Recibe un np.array con shape (n, 86, 2). Retorna un np.array con misma shape, pero con los datos normalizados.'''
        for x in X:
            self._normalize(x)
        return X

if __name__ == '__main__':
    r = Reader('all_labels.txt', race='A', sex='F')
    to_plot = 10
    X, y = r.read()
    n = Normalizer()
    print(X)
    for i, face in enumerate(X[ : to_plot]):
        print(i)
        for landmark in face:
            pl.plot(landmark[0], -landmark[1], '.')

    pl.show()
    XX = n.normalize(X)
    for i, face in enumerate(X[ : to_plot]):
        print(i)
        for landmark in face:
            pl.plot(landmark[0], -landmark[1], '.')

    pl.show()
