import pandas as pd
import prince
import matplotlib.pyplot as plt
import numpy as np

data = {
    "chatains": [119, 54, 29, 84],
    "roux": [26, 14, 14, 17],
    "blonds": [7, 10, 16, 94]
}
df = pd.DataFrame(data, index=["marrons", "noisette", "verts", "bleus"])

afc = prince.CA(n_components=2)
afc.fit(df)

coord_ind = afc.row_coordinates(df)
coord_var = afc.column_coordinates(df)

eigvals = afc.eigenvalues_
eigvects = coord_var * np.sqrt(eigvals)

plt.figure(figsize=(10, 8))
plt.scatter(coord_ind.iloc[:, 0], coord_ind.iloc[:, 1], c='blue', label='Individus')
plt.scatter(coord_var.iloc[:, 0], coord_var.iloc[:, 1], c='red', marker='x', label='Variables')
for i, ind in coord_ind.iterrows():
    plt.annotate(i, (ind[0], ind[1]), textcoords="offset points", xytext=(-10, -10), ha='center', fontsize=8,
                 color='blue')
for i, var in coord_var.iterrows():
    plt.annotate(i, (var[0], var[1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='red')

plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.legend()
plt.title('AFC')
plt.grid(True)
plt.show()