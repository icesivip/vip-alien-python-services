from Analytics.Forecasts import forecasts
from Analytics.clustering.kmeans import KMeans
from Analytics.clustering.kprototypes import KPrototypes
from flask import Flask

app = Flask(__name__)

app.add_url_rule('/', view_func=forecasts.HelloWorld)
"""
app.add_url_rule('/', view_func=KMeans.)
app.add_url_rule('/', view_func=KPrototypes.)

"""