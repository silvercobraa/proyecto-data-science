class Extractor(object):
    """Clase encargada de extraer las features de interés para la regresión. Esto puede ser distancias importantes de la cara, ratios, u otro."""
    def __init__(self):
        pass

    def extract(self, X):
        '''No implementado aún. Recibe un vector de (n, 86, 2). Retorna un vector de (n, d), donde d es la cantidad de features.'''
        return X.reshape((len(X), 172))

if __name__ == '__main__':
    # Test unitario.
    pass
