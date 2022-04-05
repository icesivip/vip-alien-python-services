from types import SimpleNamespace

from Analytics.Visualization import graphs
from Analytics.clustering.kmeans import KMeans
from Analytics.clustering.kprototypes import KPrototypes
from Analytics.clustering.Pruebas.datasets.routes import folder
from Analytics.clustering.model import controller
from Analytics.clustering.model import KMeans_Wrapper
from flask_cors import CORS
import json
import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)

app.secret_key = 'super secret key'

app.config['UPLOAD_FOLDER'] = folder

CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/fit-km')
def fit_kmeans():
    pass


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/testt', methods=['GET', 'POST'])
def test():
    return "a"


@app.route('/upl', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            try:
                k = int(request.form.__getitem__('clusters'))
            except:
                k = 3

            try:
                iteration = int(request.form.__getitem__('iteration'))
            except:
                iteration = 200

            try:

                str_model = request.form.__getitem__('model')
                print('>> ', str(str_model))
                model_dic = json.loads(str_model)

                parsed_model = json.loads(str_model, object_hook=lambda d: SimpleNamespace(**d))

                print('>>>>>> <<<<<<<')

                print('model', vars(parsed_model.clasified_data)['0'])
                model = vars(parsed_model)
            except:
                model = 0
                print('error!')

            json_res = controller.fit_data('/' + filename, k, iteration, model)

            return json_res
    return 'Oh no'


@app.route('/uplkp', methods=['GET', 'POST'])
def upload_file_kp():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            data = controller.fit_data_kp('/' + filename)

            return jsonify(data)

    return 'Oh no'


if(__name__ == '__main__'):
    app.run(debug=True)