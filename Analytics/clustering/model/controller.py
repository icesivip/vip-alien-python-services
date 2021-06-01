import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import style

from Analytics.PCA import pca
from Analytics.clustering.kmeans import KMeans
from Analytics.Visualization import graphs
from Analytics.clustering.Pruebas.datasets.routes import folder


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



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

    x = np.empty(0)
    y = np.empty(0)

    df = km.clasified_data

    dumped = json.dumps(df, cls=NumpyEncoder)

    return dumped


