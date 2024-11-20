import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression

boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")
boston.head()

boston = boston.fillna(0) #Completo los NaN con ceros para no tener inconvenientes en la regresión

x = boston[['CRIM', 'LSTAT', 'TAX',	'ZN','INDUS',	'CHAS',	'NOX',	'RM',	'AGE',	'DIS',	'RAD',	'PTRATIO',	'B'	]]
y = boston['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  #Separamos en testeo y entrenamiento

#Empleo un selector de los mejores atributos del dataset
selector = SelectKBest(f_classif, k=4)
selector.fit(x_train, y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()

#Grafico los score de cada uno de los atributos del dataset
x_indices = np.arange(x.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(x_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number: cada uno de los elementos de x")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()

#Podemos observar que de los 13 atributos que conforman las variables independientes cuatro de ellos se destacan por su score, correspondientes a las columnas 0,1,6 y 7. Es decir las variables que tienen mayor peso sobre el precio de venta de las casas son CRIM, LSTAT, NOX y RM. Vuelvo a emplear el modelo de regresión lineal multivariada realizado en el Trabajo Práctico 6 pero esta vez solo empleando los cuatro atributos mencionados, luego de aplicar la reducción de dimensionalidad.

#Regresión lineal multivariable
boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")
boston = boston.fillna(0) #Completo los NaN con ceros para no tener inconvenientes en la regresión

x = boston[['CRIM', 'LSTAT','NOX',	'RM']]
y = boston['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  #Separamos en testeo y entrenamiento
linear_regressor = LinearRegression()
fit = linear_regressor.fit(x_train, y_train)

print('R cuadrado:',fit.score(x_train, y_train))
print('Intersección de la curva:',fit.intercept_)
print('Coeficientes:',fit.coef_)

#Regresión lineal univariada
boston = pd.read_csv("https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/3_MidtermProjects/ProjectBHP/data/housing_data.csv")
boston = boston.fillna(0) #Completo los NaN con ceros para no tener inconvenientes en la regresión

y= np.array(boston['MEDV'])
x =np.array(boston['LSTAT']).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  #Separamos en testeo y entrenamiento
linear_regressor = LinearRegression()
fit = linear_regressor.fit(x_train, y_train)

print('R cuadrado:',fit.score(x_train, y_train))
print('Intersección de la curva:',fit.intercept_)
print('Coeficientes:',fit.coef_)

#Podemos concluir que si bien con los atributos más significativos se obtienen R cuadrados de 0,6 aproximadamente, lo cual es una regresión que ajusta bien al conjunto de datos, cuando teniamos todas las variables (13) este coeficiente mejoro un poco. Es decir estamos obteniendo un ajuste correcto tanto con como sin reducción de dimensionalidad. En este dataset los atributos eran 13 y 1 variable a predecir, quizás obtengamos una diferencia más clara entre los modelos con todos los features y con reducción para datasets con más columnas.