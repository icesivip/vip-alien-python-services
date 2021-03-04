import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import style

from Analytics.PCA.pca import pca

style.use('ggplot')
#\Analytics\PCA\Pca.py

data1 = pd.read_csv('Analytics/clustering/kmeans/datasets/buddymove_holidayiq.csv', header = None)
#plt.scatter(data1[0].values, data1[1].values)
'''print(data1.tail())'''
normalized_df = ((data1-data1.mean())/data1.std()).to_numpy()
'''
print('Uno\n')
print(normalized_df)
'''
#Normalizacion de datos, para que se conserve el tipo de dato
df_scaled = StandardScaler()
df_scaled = pd.DataFrame(df_scaled.fit_transform(data1),columns  = data1.columns )
'''print(df_scaled.tail())'''


"""Acontinuecion se realizara el proceso para escoger la cantidad adecuada de clusters"""
#Se crea un arreglo donde se guardaran la distorcion con la cantidad de clusters
#En terminos matematicos, se guardara la suma de las distancias al cuadrados de todos los puntos a su centro asignado
distortions = []
K = range(1,6)
class KMeans():
  
  """
    Inicializa un nuevo objeto de tipo KMeans 
    param: k - número de clusters
    param: max_iter - número máximo de iteraciones
    pre: k esté definida
    post: se crea un nuevo objeto de tipo Kmeans
  """
  def __init__(self, k, max_iter=300, random_state=19):
    self.k = k 
    self.max_iter = max_iter
    self.centroids = {} #Diccionario que almacena los puntos del dataset que serán usados como centroides.

    self.clasified_data = {} #Diccionario que almacena las listas de puntos que pertenecen a cada centroide
    self.real_crentroids = {}
    self.distortion = 0.0

  """
    Description: Define los centroides y los modifica según el promedio de los puntos asociados a ellos.
    param: data - Dataset a analizar
    pre: k debe estar definido; data debe
    post: Se definen los clusters luego de max_iter iteraciones 
  """
    
  def fit(self, data):
    
    self.centroids = {}
    self.data = data


    '''print('\n\n\n')'''
    #Se definen como centroides los primeros k elementos del dataset
    for centroid in range(self.k):
      '''print('centroide: ', data[centroid])'''
      self.centroids[centroid] = data[centroid] 
    '''print('\n\n\n')'''
    
    """
    for centroid in range(self.k):
      self.centroids[centroid] = random.choice(data)
    """
  def step(self, data):
    self.distortion = 0.0
    self.clasified_data = {}

      #Se inicializa una lista vacía para cada cluster en el diccionario de datos clasificados
    for cluster in range(self.k):
      self.clasified_data[cluster] = []

      #Relaciona los datapoints con su cluster más cercano
    for data_point in data:
        
      min_distance = self.min_distance(data, data_point) 
      self.clasified_data[min_distance[1]].append(data_point)# Agrega el datapoint al diccionario que clasifica los datos en los diferentes clusters
      
      #se suma el valor del cluster agregado elvandolo al cuadrado
      self.distortion = self.distortion + np.square(min_distance[0])


    #Se guarda una copia de los centroides anteriores
    prev_centroids = dict(self.centroids)
    self.real_crentroids = dict(self.centroids)
    '''print(self.centroids)'''
    
    #se guarda la suma en el arreglo
    


    #Se redefinen los clusters con el promedio de los puntos que pertenecen a cada agrupación
    for key in self.clasified_data:
      '''  
      print(type(key), key)
      print('count: ', self.clasified_data[key])
      '''
      self.centroids[key] = np.average(self.clasified_data[key], axis = 0)
    '''print(self.centroids)'''

    #Partimos del supuesto que los clusters son óptimos
    optimized = True

    #Comparamos los clusters anteriores con los que acabamos da calcular. 
    # Si se mueven más del rango de toleracia, continuamos con las iteraciones (max_iter)
    for c in self.centroids:

      original_centroid = prev_centroids[c]
      current_centroid = self.centroids[c]
      if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > 0.001:
        '''print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))'''
        optimized = False
    
    return optimized

  """
    Description: Distancia euclidiana entre puntos
    param: x - Punto inicial
    param: y - Punto final
    pre: Ambos puntos 'x' y 'y' están definidos
    post: Se calcula la distancia entre ambos puntos.
  """
  def euclidean_distance(self, x, y):
    return np.sqrt(np.sum(np.square(x-y)))


  def min_distance(self, data, datapoint):

    min_distance = [-1,0]

    for i in range(self.k):
            
        distance = self.euclidean_distance(self.centroids[i], datapoint)

        if min_distance[0] == -1:
            min_distance[0] = distance
            min_distance[1] = i
        else:
                
            if min_distance[0] > distance:
              min_distance[0] = distance
              min_distance[1] = i

    return min_distance


  def cost(self, x, y):

    return 


red = pca(2)

for k in  K:
  clf = KMeans(k = k)
  df_compress = red.fit(df_scaled)
  clf.fit(df_compress)
  for i in range(50):
    clf.step(clf.data)
  distortions.append(clf.distortion)
print(len(distortions))
plt.figure(figsize=(12,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


colors = 10*["g","r","c","b","k","y","m"]

for classification in clf.clasified_data:
    color = colors[classification]
    for featureset in clf.clasified_data[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, s=80, linewidths=2)

#print(clf.centroids)
for centroid in clf.real_crentroids:

  plt.scatter(clf.real_crentroids[centroid][0], clf.real_crentroids[centroid][1],
    marker="x", color="k", s=150, linewidths=5)
                
plt.show()
















