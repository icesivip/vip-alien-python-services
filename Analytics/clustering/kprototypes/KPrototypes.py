from Analytics.clustering.kmeans import KMeans

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


