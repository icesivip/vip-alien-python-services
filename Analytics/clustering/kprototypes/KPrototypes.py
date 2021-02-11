import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from Analytics.clustering.kmeans.KMeans import KMeans


data1 = pd.read_csv('Analytics/clustering/kprototypes/datasets/super.csv', sep=',')  

print(data1)

normalized_df = ((data1-data1.mean())/data1.std()).to_numpy()

class KPrototypes(KMeans):
  
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