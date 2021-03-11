import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import style

from Analytics.PCA import pca
from Analytics.clustering.kmeans import KMeans



style.use('ggplot')
#\Analytics\PCA\Pca.py

data1 = pd.read_csv('Analytics/clustering/Pruebas/datasets/buddymove_holidayiq.csv', header = None)
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

red = pca(2)

for k in K:
    clf = KMeans(k=k)
    df_compress = red.fit(df_scaled)
    clf.fit(df_compress)
    for i in range(50):
        clf.step(clf.data)
    distortions.append(clf.distortion)
print(len(distortions))
plt.figure(figsize=(12, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

colors = 10 * ["g", "r", "c", "b", "k", "y", "m"]

for classification in clf.clasified_data:
    color = colors[classification]
    for featureset in clf.clasified_data[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, s=80, linewidths=2)

# print(clf.centroids)
for centroid in clf.real_crentroids:
    plt.scatter(clf.real_crentroids[centroid][0], clf.real_crentroids[centroid][1],
                marker="x", color="k", s=150, linewidths=5)

plt.show()
