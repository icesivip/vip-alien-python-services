import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

from Analytics.clustering.kprototypes.KPrototypes import KPrototypes


style.use('ggplot')
import numpy as np
from Analytics.clustering.kmeans import KMeans

from Analytics.clustering.Pruebas.datasets.routes import super

data1 = pd.read_csv(super, sep=',')

print(data1)

normalized_df = ((data1-data1.mean())/data1.std()).to_numpy()





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