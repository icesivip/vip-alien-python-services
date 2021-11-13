from collections import namedtuple
from copy import copy

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
from Analytics.clustering.model import KMeans_Wrapper


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


class ModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, KMeans):
            return {'k': obj.k, 'centroids': obj.centroids, 'clasified_data': obj.clasified_data,
                    'real_crentroids': obj.real_crentroids, 'distortion': obj.distortion}
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def kmeans_decoder(model):
    return namedtuple('X', model.keys())(*model.values())


def fit_data(filename, k=3, ite=200, model=0):
    route = folder + filename
    data1 = pd.read_csv(route, header=None)

    df_scaled = StandardScaler()
    df_scaled = pd.DataFrame(df_scaled.fit_transform(data1), columns=data1.columns)

    steps = {}

    centroids = {}

    if model == 0:

        km = KMeans(k=k)
        comp = pca(2)
        dat = comp.fit(df_scaled)
        km.fit(dat)
        step_number = [i for i in range(0, ite, int(ite/4))]

        for i in range(ite):
            km.step(km.data)
            if i in step_number:
                # Data
                steps[i] = copy(km.clasified_data)
                # Centroids
                centroids[i] = organize_centroids(copy(km.real_crentroids))
        # Data
        steps[ite] = km.clasified_data
        # Centroids
        centroids[ite] = organize_centroids(km.real_crentroids)

    else:

        km = KMeans_Wrapper.format_Kmeans(model, df_scaled)
        km.step(km.data)
        print('>>>>>>>>>>>> Entró')

    model_json = json.dumps({'steps': steps, 'centroids': centroids}, cls=NumpyEncoder)

    return model_json

def organize_centroids(centroids):
    real_centroids = {}

    for k in centroids:
        real_centroids[k] = {"x": centroids[k][0], "y": centroids[k][1]}

    return real_centroids

def fit_data_kp(filename, k=3):
    route = folder + filename
    data1 = pd.read_csv(route, sep=',')

    normalized_df = ((data1 - data1.mean()) / data1.std()).to_numpy()

    clf = KPrototypes(cat=[1], k=2)
    clf.fit(normalized_df)

    for i in range(200):
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
