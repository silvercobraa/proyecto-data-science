# import pylab as pl
import matplotlib.pyplot as pl
import numpy as np
import numpy.linalg as la
import pandas as pd
import os
import random

def get_xy(filename):
    df = pd.read_csv(filename, header=None, delimiter=' ')
    xs = np.array(df[0]) # las coordenadas x de los landmarks de la imagen
    ys = np.array(df[1]) # las coordenadas y de los landmarks de la imagen
    return xs, 350 - ys + 1

fig = pl.figure()
ax = fig.add_subplot(111)
# ax.set_aspect('equal', 'box')
for i in range(1, 50 + 1):
    x, y = get_xy(os.path.join('landmark_txt', 'AM' + str(i) + '.txt'))
    # contorno cara
    # ax.plot(x[:22], y[:22], '-')
    # ceja izquierda
    # ax.plot(x[22:32], y[22:32], '-')
    # ceja derecha
    # ax.plot(x[32:42], y[32:42], '-')
    # ojo izquierdo
    # ax.plot(x[42:50], y[42:50], '-')
    # ojo derecho
    # ax.plot(x[50:58], y[50:58], '-')
    # pupilas
    ojo_der = np.array([[x[58]], [y[58]]])
    ojo_izq = np.array([[x[59]], [y[59]]]) - ojo_der
    # x_mean = x.mean()
    # y_mean = y.mean()
    # norm = np.sqrt(x_mean**2 + y_mean**2)
    norm = la.norm(ojo_izq)
    cos0 = ojo_izq[0][0] / norm
    sen0 = ojo_izq[1][0] / norm
    M = np.array([[cos0, sen0], [-sen0, cos0]])
    # print('antes:', ojo_izq)
    ojo_izq = M @ ojo_izq
    dist_inv = 1 / abs(0 - ojo_izq[0][0])
    N = np.array([[dist_inv, 0], [0, dist_inv]])
    # print('despues:', ojo_izq)
    # print('CARA', i)
    for j in range(len(x)):
        if (j >= 22 and j < 58): continue
        u = np.array([[x[j]], [y[j]]]) - ojo_der
        # print(M @ u)
        Mu = N @ (M @ u)
        # if j == 58: print('ojo izq:', Mu)
        # elif j == 59: print('ojo der:', Mu)
        # Mu = M @ u
        ax.plot(Mu[0], Mu[1], '.')
        # ax.plot(x[j], y[j], '.')
    # ax.plot(ojo_der[0] - ojo_der[0], ojo_der[1] - ojo_der[1], '.')
    # ax.plot(ojo_izq[0], ojo_izq[1], '.')

    # pupilas
    # ax.plot(x[58:60], y[58:60], '-')
    # nariz
    # ax.plot(x[60:73], y[60:73], '-')
    # boca
    # ax.plot(x[73:86], y[73:86], '-')
    print(i)
print(x)
print(y)
count = 1
for i, _ in enumerate(x):
    par = (x[i], y[i])
    ax.annotate(str(count), xy=par, xytext=par)
    count += 1
pl.show()
