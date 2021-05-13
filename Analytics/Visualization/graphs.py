import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json


def plot(df, centroid):


    data = [
        go.Scatter(
            x=df['x'],  # assign x as the dataframe column 'x'
            y=df['y']
        )
]

    graph_json = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json
    '''
    def show():
        self.fig.show()
    '''
