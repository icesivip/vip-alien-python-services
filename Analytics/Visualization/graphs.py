import plotly.express as px

class graphs():

    def __init__(self):
        self.fig = None

    def plot(self, x, y, centroid):
        self.fig = px.scatter(x, y, color=centroid)

    def show(self):
        self.fig.show()
