#1) Si el dataset fuera 2D sería necesario realizar una estimación previa de los parámetros antes de emplear la curva ROC. Por ejemplo se podría usar un modelo de regresión lineal con 
#80% de datos para entrenamiento y 20% para testeo. Y una vez obtenidas las variables nuevas entrenadas aplicar la curva ROC a las mismas.

#2)
import random
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
from sklearn.metrics import classification_report
pd.__version__
import altair as alt
import seaborn as sb

A =np.zeros(shape=50)
B =np.zeros(shape=50)
i = 0
while(i < 50):
    A[i]  = random.randint(1, 100)
    i+=1
i = 0
while(i < 50):
    B[i]  = random.randint(1, 100)
    i+=1
print(A)
print(B)

#Genero el dataset cuya primer columna será x (valor) y la segunda y (clase A o B). Si la clase es A el valor será 1 y si la clase es B será 0.

data = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/dataset_linear_reg%20-%20Hoja%201.csv?raw=true", sep=",")
data.columns = ['x',	'clase']

y = data['clase'].values.reshape(-1,1)
x = data['x'].values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=['clase']

modelo_1 = SVC(random_state=0).fit(x_train, y_train)
#modelo_1=LinearRegression().fit(x_train,y_train)

y_predicta=modelo_1.predict(x_test,)

print ('------------------------------------------------------------------------')
print ('CURVA ROC (receiver operating characteristic)')
print ('------------------------------------------------------------------------')

#Grafico curva ROC

RocCurveDisplay.from_estimator(modelo_1, x_test, y_test)

plt.show()

print('Reporte del clasificador')
print(classification_report(y_test, y_predicta))

#Para un entrenamiento del 80% y validación del 20% se obtiene un área bajo la curva de 0.58. Al bajar el entrenamiento a 70% el área bajo la curva baja a 0.56.

#3)
am_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/primer_dat.csv?raw=true", sep=",")
am_dataframe.columns = ['fecha',	'confirmados',	'activos',	'recuperados',	'decesos',	'descartados',	'sospechosos',	'contencion_psicologica',	'monitoreo_epidemiologico',	'aislamiento_por_contacto_estrecho',	'aprehendidos']

am_dataframe.columns
am_dataframe.head()

alt.Chart(am_dataframe).mark_line().encode(y= 'confirmados',x='fecha')
alt.Chart(am_dataframe).mark_line().encode(y= 'activos',x='fecha')
alt.Chart(am_dataframe).mark_line().encode(y= 'recuperados',x='fecha')
alt.Chart(am_dataframe).mark_line().encode(y= 'descartados',x='fecha')
alt.Chart(am_dataframe).mark_line().encode(y= 'sospechosos',x='fecha')
alt.Chart(am_dataframe).mark_line().encode(y= 'decensos',x='fecha')

#Desde 29-07-2020 a 20-08-2020 los casos activos dejan de crecer y se estacionan. Mientras, en ese intervalo los casos recuperados aumentan por lo que los activos deberían en realidad bajar y no quedarse en un valor aproximado fijo. Cuantos más recuperados menos casos activos habrá.

#La variable descensos puede considerarse normal ya que aumenta o se mantiene en el valor actual, siempre siendo menor o igual al número de casos confirmados.
