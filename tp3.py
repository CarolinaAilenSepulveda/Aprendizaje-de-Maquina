#1)
#Armo un dataset con tres features (x,y,clase) con 200 datos, 100 para cada clase (A o B). Tomo 100 datos del dataset completo para entrenar al clasificador y luego los restantes 100 para comprobar su validez. La primer columna es la clase, la segunda X y la tercera Y.
#Se ejecuta el código de abajo dos veces, la primera para obtener los 100 valores de X y luego para obtener los valores de Y.

#Librerias
import random
import numpy as np
import pandas as pd
pd.__version__
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler

A =np.zeros(shape=100)
B =np.zeros(shape=100)
i = 0
while(i < 100):
    A[i]  = random.randint(1, 100)
    i+=1
i = 0
while(i < 100):
    B[i]  = random.randint(1, 100)
    i+=1
print(A)
print(B)

entrenamiento_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/dataset_entrenamiento_knn%20-%20Hoja%201.csv?raw=true", sep=",")
entrenamiento_dataframe.columns = ['clase',	'x','y']

ax = plt.axes()
ax.scatter(entrenamiento_dataframe.loc[entrenamiento_dataframe['clase'] == 'A', 'x'],
           entrenamiento_dataframe.loc[entrenamiento_dataframe['clase'] == 'A', 'y'],
           c="red",
           label="Clase A")

ax.scatter(entrenamiento_dataframe.loc[entrenamiento_dataframe['clase'] == 'B', 'x'],
           entrenamiento_dataframe.loc[entrenamiento_dataframe['clase'] == 'B', 'y'],
           c="blue",
           label="Clase B")

plt.xlabel("x")
plt.ylabel("y")
ax.legend()
plt.show()


entrenamiento_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/dataset_entrenamiento_knn%20-%20Hoja%201.csv?raw=true", sep=",")
entrenamiento_dataframe.columns = ['clase',	'x','y']

validez_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/Dataset_validez_sinclase%20-%20Hoja%201.csv?raw=true", sep=",")
validez_dataframe.columns = ['x','y']

validez_dataframe_real = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/dataset_validez_knn%20-%20Hoja%201.csv?raw=true", sep=",")
validez_dataframe_real.columns = ['clase','x','y']

knn = KNeighborsClassifier(n_neighbors=10)
x = entrenamiento_dataframe[['x', 'y']]
y = entrenamiento_dataframe['clase']
knn.fit(x, y)
prediccion = knn.predict(validez_dataframe)

print('Clase que predijo el modelo')
print(prediccion)

print('Clase real del dataset')
print(validez_dataframe_real['clase'].values)

print('Reporte del clasificador')
print(classification_report(validez_dataframe_real['clase'].values, prediccion))

print('Score - Exactitud')
print(knn.score(x,y))

#Para un número de vecinos igual a 10 logramos una exactitud de 0.6 , precisión clase A 0.52, precisión clase B 0.53, f1-score 0.53 y support 49 y 50 respectivamente.
#Al subir el número de vecinos a 30 aumenta la precisión de ambas clases a 0.55 y 0.54, lo cual no implica una mejora significativa a los resultados anteriores.
#Considero que sucede esto aunque se aumente el número de vecinos ya que ambas clases fueron generadas a partir de un random() de 0 a 100, por lo que los datos pueden llegar a ser parecidos. A su vez, al ver el primer gráfico generado con ambas clases en diferentes colores, observamos los datos dispersos y mezclados entre si. No se observa una clara tendencia para ninguna de las clases, lo que puede generar que el clasificador no sepa como identificar correctamente los puntos y se vuelva difícil plantear una división entre A y B.

#2)

entrenamiento_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/dataset_entrenamiento_knn%20-%20Hoja%201.csv?raw=true", sep=",")
entrenamiento_dataframe.columns = ['clase',	'x','y']

validez_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/Dataset_validez_sinclase%20-%20Hoja%201.csv?raw=true", sep=",")
validez_dataframe.columns = ['x','y']

validez_dataframe_real = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/dataset_validez_knn%20-%20Hoja%201.csv?raw=true", sep=",")
validez_dataframe_real.columns = ['clase','x','y']

knn = KNeighborsClassifier(n_neighbors=10)
x = entrenamiento_dataframe[['x', 'y']]
y = entrenamiento_dataframe['clase']
knn.fit(x, y)
prediccion = knn.predict(validez_dataframe)

#Validación cruzada k fold

kfold_validacion = KFold(10)
resultados = cross_val_score(knn, x, y, cv = kfold_validacion)

print('Resultados Validación K fold')
print(resultados)

print('-----------------------------------------')
print('Promedio de los resultados de Validación K fold')
resultados.mean()

#Empleando un fold de 10 obtenemos una validación de 0.4 lo cual es bastante bajo a lo esperado, aunque según los resultados del Ejercicio 1 sabemos que no se pudo diferenciar correctamente la clase A de B, ya que no presentan tendencias en sus datos.
#Subiendo el fold a 20 sube la validación a 0.47 y bajando a 5 baja respectivamente el promedio de los resultados a 0.31.

#3)
#Desarrollar clasificador que prediga si una canción va a ser gustada o no por el usuario. El target = 1 implica que la canción fue gustada y el target = 0 que no. Tomo un total de 35 elementos del dataset para usarlos como validación y el resto como entrenamiento.

entrenamiento_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/Dataset_entrenamiento_target.csv?raw=true", sep=",")
entrenamiento_dataframe.columns= ['number','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','target','song_title','artist']

#entrenamiento_dataframe.head(10)

validez_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/Dataset_validacion_sinclase.csv?raw=true", sep=",")
validez_dataframe.columns = ['number','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','song_title','artist']

validez_dataframe_real = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/Validacion_contarget.csv?raw=true", sep=",")
validez_dataframe_real.columns = ['number','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','target','song_title','artist']

x = entrenamiento_dataframe[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']]
y = entrenamiento_dataframe['target']

x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)

#Empleando un clasificador KNN -------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=10)

x = entrenamiento_dataframe[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']]
y = entrenamiento_dataframe['target']

knn.fit(x, y)
prediccion = knn.predict(validez_dataframe.drop(['number','song_title','artist'],axis=1))

print('El modelo predijo que las canciones gustaran (1) o no (0)')
print(prediccion)

print('Resultado real del dataset')
print(validez_dataframe_real['target'].values)

print('Reporte del clasificador')
print(classification_report(validez_dataframe_real['target'].values, prediccion))

#ARBOL DE DECISION -----------------------------------------------------

#features = ['number','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']

#Árbol de decisión
#arbol = DecisionTreeRegressor(max_depth=3)
#arbol.fit(x_train, y_train)
#fig = plt.figure(figsize=(14,10))
#_ = plot_tree(arbol, feature_names=features, filled=True)

#print ('--------------------------------------------------------')
#print('R2 arbol de decision: ' + str(arbol.score(x_test, y_test)))
#print ('--------------------------------------------------------')

#Podemos ver que la precisión para la clase 0 fue 1 ya que es correcto que las canciones elegidas no gustaron (0) pero fue de 0 para la clase 1 ya que identifico algunas canciones como que si gustaron.
#Considero que quizas que el dataset de validación sea solo de 35 muestras y todas con clase real 0 no permitió analizar en más detalle el clasificador. A su vez en el análisis no se tiene en cuenta el nombre de la canción ni el artista, los cuales son determinantes en las elecciones de canciones de las personas, ya que nos da indicios de los géneros musicales que les gustan o de las bandas/artistas a los que siguen en Spotify.