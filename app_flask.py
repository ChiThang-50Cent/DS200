import pandas as pd
import json
import plotly
import plotly.express as px

from flask import Flask, render_template

from flask_app import utils_flask as f_utils
from main_utils import utils as u


app = Flask(__name__, template_folder='./flask_app/templates')

# init Spark and read data
spark, sc = u.initialize_spark()
pd_df = f_utils.read_data(spark)

string_idx, enc_m = f_utils.init_pre_model()
list_model = f_utils.init_ml_model()

@app.route("/")
def index():
    data = f_utils.create_dashboard(pd_df)
    return render_template("dashboard.html", data = data)

# @app.get('/')
# def index():
#     return render_template("dashboard.html", data = {})

@app.get('/test')
def test():
    return render_template('model.html')

if __name__ == "__main__":

    app.run(debug=True, port=5861)