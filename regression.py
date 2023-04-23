import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df=pd.read_csv("Automobile_data.csv")
#print(df)
df=df.loc[:,["highway-mpg","engine-size","horsepower","curb-weight","price"]]
df.dropna(inplace=True)
df.dropna(subset=["highway-mpg","engine-size","horsepower","curb-weight","price"],inplace=True)

#Diviser le dataset en X et Y
X=df.iloc[:, :-1].values
Y=df.iloc[:,-1].values

#Diviser le dataset entre le training set et le test set
Xtrain,Xtest,Ytrain,Ytest=train_test_split(df,test_size=0.2)


#Construction du modele
regressor =LinearRegression()
regressor.fit(Xtrain,Ytrain)

#Faire de nouvelles predictions
Ypred =regressor.predict(Xtest)
print(regressor.predict(np.array([[15]])))

#visualiser les resultats
plt.scatter(Xtest,Ytest,color= 'blue')
plt.plot(Xtrain,regressor.predict(Xtrain),color= 'red')
plt.xlabel("F1 ") # modification du nom de l'axe X
plt.ylabel("F2") # idem pour axe Y
plt.title("figure ")
plt.show()

