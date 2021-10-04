import numpy as np
from Analytics.clustering.kmeans.KMeans import KMeans


def format_Kmeans(kmeans, data):
    model = KMeans(int(kmeans['k']))
    model.data = data

    model.centroids = parse_centroids(vars(kmeans['centroids'], type=np.float64))
    print(kmeans['centroids'])
    print(model.centroids)
    model.clasified_data = parse_clasified_data(vars(kmeans['clasified_data']))
    model.distortion = np.float64(kmeans['distortion'])

    return model


def parse_centroids(centroids):
    dic = centroids.copy()
    for i in centroids.keys():
        key = int(i)
        dic[key] = np.array(centroids[i], dtype=np.float64)
        del dic[i]

    return dic


def parse_clasified_data(clasified_data):
    dic = clasified_data.copy()
    for i in clasified_data.keys():
        key = int(i)
        dic[key] = [np.array(x, dtype=np.float64) for x in clasified_data[i]]
        del dic[i]

    return dic
