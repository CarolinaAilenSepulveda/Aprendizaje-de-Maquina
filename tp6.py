#Ejercicio 6.1: Boston Housing Prices: El dataset contiene 506 observaciones de 14
#variables y precios de venta de propiedades en la ciudad de Boston. El
#objetivo es conocer la influencia de cada variable y elaborar un modelo que
#prediga precios de venta.

#Librerias
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import altair as alt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split

boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")
boston.head()

#Busco a que se referencian las columnas del dataset.

#CRIM - per capita crime rate by town

#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.

#INDUS - proportion of non-retail business acres per town.

#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)

#NOX - nitric oxides concentration (parts per 10 million)

#RM - average number of rooms per dwelling

#AGE - proportion of owner-occupied units built prior to 1940

#DIS - weighted distances to five Boston employment centres

#RAD - index of accessibility to radial highways

#TAX - full-value property-tax rate per $10,000

#PTRATIO - pupil-teacher ratio by town

#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

#LSTAT - % lower status of the population

#MEDV - Median value of owner-occupied homes in $1000's

#Regresión de una variable con independiente CRIM y dependiente el precio de los hogares
boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")
boston = boston.fillna(0) #Completo los NaN con ceros para no tener inconvenientes en la regresión

#Variables para la regresión lineal
y= np.array(boston['MEDV'])
x =np.array(boston['CRIM']).reshape(-1, 1)

modelo = LinearRegression()
modelo.fit(x, y)

print ("Termino independiente:",modelo.intercept_)
print("R cuadrado:",modelo.score(x,y))
alt.Chart(boston).mark_line().encode(y= 'MEDV',x='CRIM')

sns.set(style="ticks", color_codes=True)

boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")

#Primero visualizamos las diferentes variables que influyen en los precios de las casas
sns.pairplot(boston, x_vars=['CRIM','ZN','INDUS'], y_vars='MEDV',  height=3, aspect=1., kind='reg')
sns.pairplot(boston, x_vars=['CHAS','NOX','RM'], y_vars='MEDV',  height=3, aspect=1., kind='reg')
sns.pairplot(boston, x_vars=['AGE','DIS','RAD'], y_vars='MEDV',  height=3, aspect=1., kind='reg')
sns.pairplot(boston, x_vars=['TAX','PTRATIO','B','LSTAT'], y_vars='MEDV',  height=3, aspect=1., kind='reg')

#Regresión lineal multivariable
boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")
boston = boston.fillna(0) #Completo los NaN con ceros para no tener inconvenientes en la regresión

x = boston[['CRIM', 'LSTAT', 'TAX',	'ZN','INDUS',	'CHAS',	'NOX',	'RM',	'AGE',	'DIS',	'RAD',	'PTRATIO',	'B'	]]
y = boston['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  #Separamos en testeo y entrenamiento
linear_regressor = LinearRegression()
fit = linear_regressor.fit(x_train, y_train)

print('R cuadrado:',fit.score(x_train, y_train))
print('Intersección de la curva:',fit.intercept_)
print('Coeficientes:',fit.coef_)

#Podemos observar que con la regresión lineal multivariada el coeficiente R cuadrado da un resultado bueno de 0,7 muy cercano al ajuste perfecto 1. A comparación del coeficiente encontrado con una sola variable independiente de 0,15 este modelo mejora notablemente.