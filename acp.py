import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Automobile_data.csv')
df=df.loc[:,['curb-weight','engine-size','horsepower','highway-mpg','price']]
df.dropna(inplace=True)
df.dropna(subset=['curb-weight','engine-size','horsepower','highway-mpg','price'],inplace=True)
xcr=(df-df.mean()) / df.std()
matrice=xcr.corr()
matrice=matrice.values
valp,vectp =np.linalg.eig(matrice)
vectp = np.transpose(vectp)
proj1=np.matmul(matrice,vectp[0])
proj2=np.matmul(matrice,vectp[1])
data = pd.DataFrame({'proj1': proj1,
                   'proj2': proj2})
data.plot.scatter(x='proj1',y='proj2')
#plt.show()
# Tracer le cercle de corrélation
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('proj1')
ax.set_ylabel('proj2')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

for i, (x, y) in enumerate(zip(vectp[0, :], vectp[1, :])):
    ax.text(x, y, df.columns[i], ha='center', va='center', fontsize=10)
    ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='red', ec='red', width=0.005)

plt.title('Cercle de Corrélation')
plt.show()



















