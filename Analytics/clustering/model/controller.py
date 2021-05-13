import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import style

from Analytics.PCA import pca
from Analytics.clustering.kmeans import KMeans
from Analytics.Visualization import graphs
from Analytics.clustering.Pruebas.datasets.routes import folder


def fit_data(filename, k=3):
    route = folder + filename
    data1 = pd.read_csv(route, header=None)

    df_scaled = StandardScaler()
    df_scaled = pd.DataFrame(df_scaled.fit_transform(data1), columns=data1.columns)

    km = KMeans(k=k)
    comp = pca(2)
    dat = comp.fit(df_scaled)
    km.fit(dat)
    for i in range(200):
        km.step(km.data)

    x = np.array()
    y = np.array()

    for classification in km.clasified_data:

        for featureset in km.clasified_data[classification]:
            x.append(featureset[0])
            y.append(featureset[1])

    df = pd.DataFrame({'x': x, 'y': y})

    return df
