#A partir de los cuatro datasets presentados y sabiendo por enunciado que los parámetros estadísticos de los cuatro son iguales debemos encontrar otra forma de analizar si los puntos (16,8)(8,6)(16,10)(16,6) pertenecen o no a un dataset determinado. Para ello decido graficar los cuatro datasets más los cuatro valores de incógnita para analizar si siguen o no la tendencia general de las curvas.

import pandas as pd
pd.__version__
import altair as alt
import seaborn as sb
import matplotlib.pyplot as plot


am_dataframe = pd.read_csv("https://github.com/CarolinaAilenSepulveda/AM/blob/main/Datasets%20-%20Hoja%201.csv?raw=true", sep=",")
am_dataframe.columns = ['x1','y1','x2','y2','x3','y3','x4','y4']

am_dataframe.columns
am_dataframe.head()

alt.Chart(am_dataframe).mark_point().encode(y='y1',x='x1').interactive()

#Con este gráfico podemos observar que el punto (8,6), (16,10) (16,8) se encuentran en la tendencia general de los datos del dataset original. Luego analizando los próximos datasets podremos ver cual de los tres es el que mejor ajusta.


alt.Chart(am_dataframe).mark_point().encode(y='y2',x='x2').interactive()

#Podemos ver que el dato que mejor ajusta a la curva es (16,6) por lo tanto pertenece a este dataset.


alt.Chart(am_dataframe).mark_point().encode(y='y3',x='x3').interactive()

#Podemos observar que el dato que mejor ajusta a la tendencia, de tipo lineal creciente, es (16,10) por lo tanto pertenece a este dataset.

alt.Chart(am_dataframe).mark_point().encode(y='y4',x='x4').interactive()

#El dato que mejor ajusta en este caso es (8,6) ya que todos los datos del dataset menos uno tienen como componente x el valor 8 por lo tanto esta será la condición necesaria para pertenecer a la tendencia.

#Como conclusión del ejercicio, los puntos pertenecen a los siguientes datasets:
#(16,8)  - DATASET 1
#(8,6)   - DATASET 4
#(16,10) - DATASET 3
#(16,6)  - DATASET 2




