from Forecasts import forecasts
from flask import Flask

app = Flask(__name__)

app.add_url_rule('/', view_func=forecasts.HelloWorld)

