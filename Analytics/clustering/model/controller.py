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
from Analytics.clustering.kprototypes.KPrototypes import KPrototypes


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

def fit_data_kp(filename, k=3):
    route = folder + filename
    data1 = pd.read_csv(route, sep=',')

    print(data1)

    normalized_df = ((data1 - data1.mean()) / data1.std()).to_numpy()

    clf = KPrototypes(cat=[1], k=3)
    clf.fit(normalized_df)

    for i in range(50):
        clf.step(clf.data)

    df_kp = pd.DataFrame()

    for classification in clf.clasified_data:
        df_temp = pd.DataFrame(clf.clasified_data[classification])
        df_kp = df_kp.append(df_temp)

    print(clf.clasified_data)

    comp = pca(2)

    dat = comp.fit(df_kp)

    count = 0
    print(clf.clasified_data)

    blist = []

    for classification in clf.clasified_data:
        list = []
        for featureset in clf.clasified_data[classification]:
            temp_dic = {}
            temp_dic['x'] = dat[count][0]
            temp_dic['y'] = dat[count][1]
            list.append(temp_dic)
            count += 1

        blist.append(list)

    return blist
