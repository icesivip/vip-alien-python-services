import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


data1 = pd.read_csv('Analytics/clustering/kprototypes/datasets/super.csv', sep=',')  

print(data1)

normalized_df = ((data1-data1.mean())/data1.std()).to_numpy()



class KPrototypes():
  
  """
    Inicializa un nuevo objeto de tipo KMeans 
    param: k - número de clusters
    param: max_iter - número máximo de iteraciones
    pre: k esté definida
    post: se crea un nuevo objeto de tipo Kmeans
  """
  def __init__(self, k, cat, plot_var, max_iter=300):
    self.k = k 
    self.max_iter = max_iter
    self.centroids = {} #Diccionario que almacena los puntos del dataset que serán usados como centroides.

    self.categorical = cat #Índices de las variables categóricas
    
    self.plot_var = plot_var
    self.data = 0 
    self.clasified_data = {} #Diccionario que almacena las listas de puntos que pertenecen a cada centroide
  


  """
    Description: Define los centroides y los modifica según el promedio de los puntos asociados a ellos.
    param: data - Dataset a analizar
    pre: k debe estar definido; data debe
    post: Se definen los clusters luego de max_iter iteraciones 
  """
  def fit(self, data):
    
    self.centroids = {}
    self.data = data

    #Se definen como centroides los primeros k elementos del dataset
    for centroid in range(self.k):
      self.centroids[centroid] = data[centroid] 

  def step(self, data):

    self.clasified_data = {}

      #Se inicializa una lista vacía para cada cluster en el diccionario de datos clasificados
    for cluster in range(self.k):
      self.clasified_data[cluster] = []

      #Relaciona los datapoints con su cluster más cercano
    for data_point in data:
        
      min_distance = self.min_distance(data, data_point) 
      self.clasified_data[min_distance[1]].append(data_point) # Agrega el datapoint al diccionario que clasifica los datos en los diferentes clusters
        
      #Se guarda una copia de los centroides anteriores
      prev_centroids = dict(self.centroids)

      #Se redefinen los clusters con el promedio de los puntos que pertenecen a cada agrupación
      for key in self.clasified_data:
        
        #print(type(key), key)
        self.centroids[key] = np.average(self.clasified_data[key], axis = 0)
        
      #Partimos del supuesto que los clusters son óptimos
      optimized = True

      #Comparamos los clusters anteriores con los que acabamos da calcular. 
      # Si se mueven más del rango de toleracia, continuamos con las iteraciones (max_iter)
      for c in self.centroids:

        original_centroid = prev_centroids[c]
        current_centroid = self.centroids[c]
        if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > 0.001:
          #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
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


  def dissimilarities_function(self, x, y):

    cost = 0
    for j in range(0, len(x)):

      if(x[j] != y[j]):
        cost += 1

    return cost
    

  def cost(self, x, y):

    columns =  self.data.shape[-1]
    numerical = list(set( range(0,columns) ) - set(self.categorical))

    return self.dissimilarities_function(x[self.categorical], y[self.categorical]) + self.euclidean_distance(x[numerical], y[numerical])


  def min_distance(self, data, datapoint):

    min_distance = [0,0]

    for i in range(self.k):
            
        distance = self.cost(self.centroids[i], datapoint)

        if min_distance[0] == 0:
            min_distance[0] = distance
            min_distance[1] = i
        else:
                
            if min_distance[0] > distance:
              min_distance[0] = distance
              min_distance[1] = i

    return min_distance


"""
clf = KPrototypes(cat = [0, 1], plot_var= [3,4], k = 6)
clf.fit(normalized_df)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="x", color="k", s=150, linewidths=5)

colors = 10*["g","r","c","b","k"]

for classification in clf.clasified_data:
    color = colors[classification]
    for featureset in clf.clasified_data[classification]:
        plt.scatter(featureset[clf.plot_var[0]], featureset[clf.plot_var[1]], color=color, s=80, linewidths=2)

plt.show()
"""