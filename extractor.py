class Extractor(object):
    """Clase encargada de extraer las features de interés para la regresión. Esto puede ser distancias importantes de la cara, ratios, u otro."""
    def __init__(self):
        pass

    def extract(self, X):
        '''No implementado aún. Recibe un vector de (n, 86, 2). Retorna un vector de (n, d), donde d es la cantidad de features.'''
        for i in range(X.shape[0]):
        	print('*'*60)
        	#Ratio 1: mideye distance to interocular distance
        	mideye_dist = np.linalg.norm(X[i][59] - X[i][58])
        	interocular_dist = np.linalg.norm(X[i][54] - X[i][42])
        	ratio_1 = mideye_dist / interocular_dist
        	#print('Ratio 1: ', ratio_1, ' de ', i)

        	#Ratio 2: mideye distance to nose width
        	nose_width = np.linalg.norm(X[i][64] - X[i][68])
        	ratio_2 = mideye_dist / nose_width
        	#print('Ratio 2: ', ratio_2, ' de ', i)

        	#Ratio 3: Mouth width to Interocular distance
        	mouth_width = np.linalg.norm(X[i][79] - X[i][73])
        	ratio_3 = mouth_width / interocular_dist
        	#print('Ratio 3: ', ratio_3, ' de ', i)

        	#Ratio 4: Lips-chin distance to interocular distance
        	lipsChin_dist = np.linalg.norm(X[i][76] - X[i][11])
        	ratio_4 = lipsChin_dist / interocular_dist
        	#print('Ratio 4: ', ratio_4, ' de ', i)

        	#Ratio 5: Lips-chin distance to noise width
        	ratio_5 = lipsChin_dist / nose_width
        	#print('Ratio 5: ', ratio_5, ' de ', i)

        	#Ratio 6: Interocular distance to eye fissure width
        	eyeFiss_width = np.linalg.norm(X[i][54] - X[i][50])
        	ratio_6 = interocular_dist / eyeFiss_width
        	#print('Ratio 6: ', ratio_6, ' de ', i)

        	#Ratio 7: Interocular distance to lip height
        	lip_height = np.linalg.norm(X[i][76] - X[i][82])
        	ratio_7 = interocular_dist / lip_height
        	#print('Ratio 7: ', ratio_7, ' de ', i)

        	#Ratio 8: Nose width to eye fissure width
        	ratio_8 = nose_width / eyeFiss_width
        	#print('Ratio 8: ', ratio_8, ' de ', i)

        	#Ratio 9: Nose width to lip height
        	ratio_9 = nose_width / lip_height
        	#print('Ratio 9: ', ratio_9, ' de ', i)

        	#Ratio 10: Lip height to nose-mouth distance
        	noseMouth_dist = np.linalg.norm(X[i][66] - X[i][82])
        	ratio_10 = lip_height / noseMouth_dist
        	#print('Ratio 10: ', ratio_10, ' de ', i)

        	#Ratio 11: Length of face to width of face
        	length_face = np.linalg.norm(X[i][0] - X[i][11])
        	width_face = np.linalg.norm(X[i][4] - X[i][18])
        	ratio_11 = length_face / width_face
        	#print('Ratio 11: ', ratio_11, ' de ', i)

        	#Ratio 12: Nose-chin distance to lips-chin distance
        	noseChin_dist = np.linalg.norm(X[i][66] - X[i][11])
        	ratio_12 = noseChin_dist / lipsChin_dist
        	#print('Ratio 12: ', ratio_12, ' de ', i)

        	#Ratio 13: Nose width to nose-mouth distance
        	ratio_13 = nose_width / noseMouth_dist

        	#Ratio 14: Mouth width to nose width
        	ratio_14 = mouth_width / nose_width

        	#Ratio 15: Length of nose to nose-chin distance
        	midUp_nose = X[i][72] - X[i][60] / 2
        	length_nose = np.linalg.norm(midUp_nose - X[i][66])
        	ratio_15 = length_nose / noseChin_dist

        	#Ratio 16: Interocular distance to nose width
        	ratio_16 = interocular_dist / nose_width

        	#Ratio 17: Length of face to 4 times nose width
        	ratio_17 = length_face / (4 * nose_width)

        return X.reshape((len(X), 172))

if __name__ == '__main__':
    # Test unitario.
    pass
