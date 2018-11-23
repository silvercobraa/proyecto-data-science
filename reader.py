import os
import pandas as pd
import numpy as np

class Reader(object):
    """
    Clase encargada de leer el dataset.
    race: indica la raza a considerar. Valores posibles: 'A', 'C', None (todas).
    sex: indica el sexo a considerar. Valores posibles: 'F', 'M', None (todos).
    labels_file: ruta del archivo con los puntajes de cada imagen. Default: 'train_test_files/All_labels.txt'.
    landmark_dir: ruta del directorio con los archivos landmark_txt. Default: 'landmark_txt'.
    """
    def __init__(self, labels_file=os.path.join('train_test_files', 'All_labels.txt'), landmark_dir='landmark_txt', race=None, sex=None):
        self.labels_file = labels_file
        self.landmark_dir = landmark_dir
        self.race = race
        self.sex = sex

    def _filter(self, image_name):
        """Retorna true si la imagen se debe procesar, de acuerdo a los parámetros ingresados en el constructor. Retorna falso en otro caso."""
        race = image_name[0]
        sex = image_name[1]
        return (self.race is None or self.race == race) and (self.sex is None or self.sex == sex)

    def _read_txt(self, image_name):
        """Recibe el nombre de un archivo jpg. Retorna una matriz numpy donde cada fila es un landmark (par de puntos), es decir una matriz de n x 2"""
        print('leyendo:', image_name)
        txt_name = image_name[:-3] + 'txt'
        txt_name = str(os.path.join(self.landmark_dir, txt_name))
        # print('leyendo', txt_name)
        df = pd.read_csv(txt_name, header=None, delimiter=' ')
        return df.values

    def read(self):
        df = pd.read_csv(self.labels_file, header=None, delimiter=' ')
        images, scores = df[0], df[1]
        filter = df[0].apply(self._filter)
        dff = df[filter]
        # x = dff[0].apply(self._read_txt)
        return np.array([self._read_txt(x) for x in dff[0]]), dff[1].values


if __name__ == '__main__':
    file = os.path.join('train_test_files', 'All_labels.txt')
    races = ['C', 'A']
    sexes = ['F', 'M']
    for race in races:
        for sex in sexes:
            reader = Reader(file, race=race, sex=sex)
            x, y = reader.read()
            print(x)
            print(y)
            print(x.shape, y.shape)
