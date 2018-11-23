import numpy as np

class Offset(object):
    '''Offsets del primer y último landmark de cada órgano de la cara para cada rostro. Cada órgano se nombra como ORGANO_BEGIN o ORGANO_END, indicado el indice del primer landmark de dicho órgano y el índice del landmark siguiente al último, respectivamente (intervalo semi abierto, para facilitar la iteracion). RIGHT y LEFT es con respecto a la imagen original y no como aparece en el plot.'''
    FACE_BEGIN = 0
    FACE_END = 22
    RIGHT_BROW_BEGIN = 22
    RIGHT_BROW_END = 32
    LEFT_BROW_BEGIN = 32
    LEFT_BROW_END = 42
    RIGHT_EYE_BEGIN = 42
    RIGHT_EYE_END = 50
    LEFT_EYE_BEGIN = 50
    LEFT_EYE_END = 58

    RIGHT_PUPIL = 58
    LEFT_PUPIL = 59

    NOSE_BEGIN = 60
    NOSE_END = 73
    MOUTH_BEGIN = 73
    MOUTH_END = 86
    def __init__(self):
        raise Exception('Clase estática, no instanciar')


if __name__ == '__main__':
    test = np.empty((86, 2))
    print(test)
    print(test[Offset.FACE_BEGIN : Offset.FACE_END, :])
    print(test[Offset.RIGHT_BROW_BEGIN : Offset.RIGHT_BROW_END, :])
    print(test[Offset.LEFT_BROW_BEGIN : Offset.LEFT_BROW_END, :])
    print(test[Offset.RIGHT_EYE_BEGIN : Offset.RIGHT_EYE_END, :])
    print(test[Offset.LEFT_EYE_BEGIN : Offset.LEFT_EYE_END, :])
    print(test[Offset.NOSE_BEGIN : Offset.NOSE_END, :])
    print(test[Offset.MOUTH_BEGIN : Offset.MOUTH_END, :])
