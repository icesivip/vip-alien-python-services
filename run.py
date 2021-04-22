from Analytics.Forecasts import forecasts
from Analytics.Visualization import graphs
from Analytics.clustering.kmeans import KMeans
from Analytics.clustering.kprototypes import KPrototypes
from flask import Flask

app = Flask(__name__)

@app.route('/fit-km', metho)
def fit_kmeans():


