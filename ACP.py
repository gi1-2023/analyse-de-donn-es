import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA


df=pd.read_csv("Automobile_data.csv")
df=df.loc[:,["highway-mpg","engine-size","horsepower","curb-weight","price"]]
df.dropna(inplace=True)
df.dropna(subset=["highway-mpg","engine-size","horsepower","curb-weight","price"],inplace=True)

XCR=(df-df.mean())/df.std()
matrice=XCR.corr()
valp,vecteur=np.linalg.eig(matrice)
principe1=np.matmul(matrice,vecteur[0])
principe2=np.matmul(matrice,vecteur[1])
matrice_corr= pd.DataFrame({'F1': principe1,
                                'F2': principe2})

#le nuage des points
matrice_corr.plot.scatter('F1','F2')
plt.xlabel("F1 (16%)") # modification du nom de l'axe X
plt.ylabel("F2( 16%)") # idem pour axe Y
plt.suptitle("nuages des points ") # titre général
plt.show()

# Tracer le cercle de corrélation
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('proj1')
ax.set_ylabel('proj2')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

for i, (x, y) in enumerate(zip(vecteur[0, :], vecteur[1, :])):
    ax.text(x, y, df.columns[i], ha='center', va='center', fontsize=10)
    ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='red', ec='red', width=0.005)

plt.title('Cercle de Corrélation')
plt.show()

np.random.seed(42)
dr=pd.DataFrame(np.random.rand((10,3),columns=['X','Y','Z']))
pca =PCA(n_components=2)
pca.fit(dr)
varience=pca.explained_variance_ratio_
projection_qualite=sum(varience)
print("la qualite de projection est :",projection_qualite)





