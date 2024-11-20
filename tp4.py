#Ejercicio 4.4 : Winsconsin Breast Cancer dataset. Contiene una tabla con 569 datos de histología mamaria humana, 357 casos de tejido tumoral benigno y 212 tejido maligno.
#Los datos son 32 atributos radiómicos (tamaños, formas, texturas, etc.). 
#El objetivo es desarrollar un clasificador que a través de los  atributos permita predecir la benignidad o  malignidad de un tumor.

#Librerias
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns

cancer_tissues = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectBCD/data/data.csv")
cancer_tissues.head()


data_original = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectBCD/data/data.csv")
data = data_original.fillna(0) #Reemplazo los NaN por 0, como todos los campos son decimales sirve el remplazo en todos los registros 


ax = plt.axes()
ax.scatter(data.loc[data['diagnosis'] == 'B', 'radius_mean'],
           data.loc[data['diagnosis'] == 'B', 'texture_mean'],
           c="red",
           label="Benigno")

ax.scatter(data.loc[data['diagnosis'] == 'M', 'radius_mean'],
           data.loc[data['diagnosis'] == 'M', 'texture_mean'],
           c="blue",
           label="Maligno")

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

ax.legend()
plt.show()

#Podemos observar con solo dos variables que la clase Benigno se corresponde a los valores más bajos de la media de radio y textura del tejido mamario, mientras que los casos Malignos presentan mayor área y textura. El parámetro que más divide gráficamente ambos grupos es la media del radio.
#El primer clasificador que empleo es RandomForest con un 80% de los datos para entrenamiento y el 20% para testeo. En este caso se realiza una imputación de 0 para los campos NaN.

data_original = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectBCD/data/data.csv")
data = data_original.fillna(0) #Reemplazo los NaN por 0, como todos los campos son decimales sirve el remplazo en todos los registros 

y = data['diagnosis']   #Variable que quiero predecir, si es benigno o maligno
x = data.drop(['diagnosis','id'],axis=1) #Variables que no empleo para predecir
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

print ('Distribucion original:',Counter(y_train))

#Random forest
modeloforest_1=RandomForestClassifier(n_estimators=100).fit(x_train,y_train)
y_predicta_1=modeloforest_1.predict(x_test)
#print ('Valores de clase predicta:',y_predicta_1)
print ('---------------------------------------------------------------')
print ('RESULTADOS DEL CLASIFICADOR')
print ('---------------------------------------------------------------')
print ('Importancia de las variables:',modeloforest_1.feature_importances_)
print ('---------------------------------------------------------------')
print('Reporte del clasificador')
print(classification_report(y_test, y_predicta_1))

print ('MATRIZ DE CONFUSIÓN')
print ('---------------------------------------------------------------')
#Grafico matriz confusion
mc=confusion_matrix(y_test,y_predicta_1)
ax3=plt.subplot()
sns.heatmap(mc,annot=True,ax=ax3)
ax3.set_xlabel('Prediccion');ax3.set_ylabel('Valor real')
ax3.set_title('Matriz de confusion')
ax3.xaxis.set_ticklabels(['Benigno', 'Maligno']); ax3.yaxis.set_ticklabels(['Benigno', 'Maligno'])


#Viendo la matriz de confusión del clasificador observamos que para el valor real Maligno la predicción coincide 45 veces mientras que 2 se equivoca. Por otra parte para el valor real Benigno 65 veces lo precide correctamente y 2 veces incorrecto. Por lo tanto podemos concluir que el clasificador funciona de manera correcta.
#Por otra parte, contamos con el reporte del clasificador para analizar los resultados. Podemos observar que la precisión de ambas clases ronda el valor 0,9, lo cual nos indica un buen funcionamiento en ambas clases M y B. A su vez viendo los valores de f1-score y recall ambos poseen valores cercanos a 1, el caso ideal del clasificador.