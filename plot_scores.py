import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv('both.txt')
barWidth = 0.25
cats = df['category'].unique()
x_ridge = np.arange(len(cats))
x_svr = [x + barWidth for x in x_ridge]
x_rf = [x + barWidth for x in x_svr]
ridge = df[df['model'] == 'ridge']
svr = df[df['model'] == 'svr']
rf = df[df['model'] == 'rf']
pl.bar(x_ridge, ridge['mean'], width=barWidth, yerr=ridge['std'], label='Ridge')
pl.bar(x_svr, svr['mean'], width=barWidth, yerr=svr['std'], label='SVR')
pl.bar(x_rf, rf['mean'], width=barWidth, yerr=rf['std'], label='RF')
pl.xticks([r + barWidth for r in range(len(x_ridge))], cats)
pl.xlabel('Categorías', fontweight='bold')
pl.ylabel('R²', fontweight='bold')
pl.legend()
pl.show()
