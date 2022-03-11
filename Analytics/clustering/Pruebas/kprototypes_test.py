import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler

from Analytics.PCA import pca
from Analytics.clustering.kprototypes.KPrototypes import KPrototypes

style.use('ggplot')

from Analytics.clustering.Pruebas.datasets.routes import super

data1 = pd.read_csv(super, sep=',')

normalized_df = ((data1 - data1.mean()) / data1.std()).to_numpy()

clf = KPrototypes(cat=[1], k=2)
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

print(dat)

colors = 10 * ["g", "r", "c", "b", "k"]

'''
for datapoint in dat:
    plt.scatter(datapoint[0], datapoint[1], color='g', s=80, linewidths=2)
'''

count = 0
print(clf.clasified_data)


blist = []

for classification in clf.clasified_data:
    color = colors[classification]

    list = []

    for featureset in clf.clasified_data[classification]:

        temp_dic = {}

        temp_dic['x'] = dat[count][0]
        temp_dic['y'] = dat[count][1]

        list.append(temp_dic)

        plt.scatter(dat[count][0], dat[count][1], color=color, s=80, linewidths=2)
        count += 1

    blist.append(list)

print(blist)
print(len(blist[1]))


plt.show()
