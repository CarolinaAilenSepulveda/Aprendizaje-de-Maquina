#El dataset contiene registros de crímenes ocurridos durante 1973 en EEUU en diferentes
#estados, así como el porcentaje de la población que vive en zonas
#urbanas en dichos estados. El objetivo es agrupar a estos estados en diferentes categorías de acuerdo a estos factores.

#Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sb
from sklearn.metrics import pairwise_distances_argmin_min
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

violent_crime = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectUSA/data/violent_crime.csv")
violent_crime.head()

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

violent_crime = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectUSA/data/violent_crime.csv")
violent_crime.columns = ['estado','Murder','Assault','UrbanPop','Rape']

#Grafico con pairplot para ver algunas relaciones entre variables
sb.pairplot(violent_crime, hue='estado',height=5,vars=['Murder',"Assault","UrbanPop",'Rape'],kind='scatter')

violent_crime = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectUSA/data/violent_crime.csv")
violent_crime.columns = ['estado','Murder','Assault','UrbanPop','Rape']

X = np.array(violent_crime[["Rape","Assault","UrbanPop",'Murder']])
y = np.array(violent_crime['estado'])

#Empleo Kmeans con 5 clusters o agrupamientos
kmeans = KMeans(n_clusters=5,n_init=10).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

plt.rcParams['figure.figsize'] = (5,4)
plt.style.use('ggplot')

violent_crime = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectUSA/data/violent_crime.csv")
violent_crime.columns = ['estado','Murder','Assault','UrbanPop','Rape']

X = np.array(violent_crime[["Rape","Assault","UrbanPop",'Murder']])
y = np.array(violent_crime['estado'])

#Empleo Kmeans con 5 clusters o agrupamientos
kmeans = KMeans(n_clusters=5,n_init=10).fit(X)
centroids = kmeans.cluster_centers_

#Graficamos los clusters, en este código encontre la forma de marcar los centros de los clusters con una estrella del color asignado
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])

print('Murder - Assault')
fig = plt.figure()

#Plotear separados los resultados
f1 = violent_crime['Assault'].values
f2 = violent_crime['Murder'].values
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 3], marker='*', c=colores, s=1000)
plt.show()

print('Murder - UrbanPop')
fig = plt.figure()

#Plotear separados los resultados
f1 = violent_crime['UrbanPop'].values
f2 = violent_crime['Murder'].values
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 2], C[:, 3], marker='*', c=colores, s=1000)
plt.show()

print('Rape - UrbanPop')
fig = plt.figure()

#Plotear separados los resultados
f1 = violent_crime['UrbanPop'].values
f2 = violent_crime['Rape'].values
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 2], C[:, 0], marker='*', c=colores, s=1000)
plt.show()

#Podemos ver que los clusters más diferenciados y por lo tanto que ayudan a segmentar las diferentes clases dentro del dataset son los correspondientes a Murder Assault. La relación es a mayor casos de agresión/robo mayor número de asesinatos.

