import numpy as np
import json

class KMeans():
  
  """
    Inicializa un nuevo objeto de tipo KMeans 
    param: k - número de clusters
    param: max_iter - número máximo de iteraciones
    pre: k esté definida
    post: se crea un nuevo objeto de tipo Kmeans
  """
  def __init__(self, k, random_state=19):
    self.k = k
    self.centroids = {} #Diccionario que almacena los puntos del dataset que serán usados como centroides.
    self.data = 0
    self.clasified_data = {} #Diccionario que almacena las listas de puntos que pertenecen a cada centroide
    self.stats_data = {} #Diccionario que almacena las estadisticas (cuenta, porcentaje del total, varianza,desviacion estander) que pertenecen a cada centroide
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
    self.stats_data = {}
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

    self.calculate_stats()
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

  def calculate_stats(self):
    data = self.clasified_data
    totalAmmount = self.data.count #cantidad de datos total
    for centroid in data:
      values  = data[centroid]
      ammount = values.count
      percent = ammount/totalAmmount
      variance = values.var()
      desviation = values.std()

      self.stats_data[centroid] = np.array([ammount, percent, variance, desviation], float)

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


  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
















