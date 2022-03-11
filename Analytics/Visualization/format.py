import pandas as pd
from sklearn.preprocessing import StandardScaler

from Analytics.PCA import pca


def format_data(data):

    df_scaled = StandardScaler()
    df_scaled = pd.DataFrame(df_scaled.fit_transform(data), columns=data.columns)

    p = pca(2)
    formatted_data = p.fit(df_scaled)

    return formatted_data
