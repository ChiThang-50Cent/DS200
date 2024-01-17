# to new branch
import json
import numpy as np
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

import plotly
import plotly.express as px

from pyspark.sql.types import *
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, IsotonicRegressionModel, FMRegressionModel
from pyspark.ml.feature import OneHotEncoderModel
from pyspark.ml import PipelineModel

from main_utils import utils as u
from main_utils.crawl_url import *
from main_utils.crawl_data import *
from main_utils.clean_data import *
from main_utils.train_model import *
from main_utils.feature_extract import *

def tranformFetures(X, string_idx, enc_m, use_transform=True):
    ###########################

    if use_transform:
        X = typeCasting(X)
        X = from_pd_to_spark(X)

    st.write(X)
    # st.write(X.head().TongGia)

    scaled_X = featureExtraction(X, string_idx, enc_m)
    ###########################

    return scaled_X

def prediction(samples, model, use_transform=True):
    
    X = tranformFetures(samples, use_transform=use_transform)
    pred = model.predict(X.head().features)

    # Lấy kết quả dự đoán.
    # results = pd.DataFrame({'Giá dự đoán': [pred]})

    # Test 
    results = get_result(X.head().TongGia, pred)
    
    # Xuất ra màn hình
    return results

def create_dashboard(df):
    numOfProject = df.shape[0]
    meanPrice = round(df['TongGia'].mean() * 1000000)

    #Figure 1
    fig1 = px.histogram(df, x="Tinh", color="LoaiBDS", labels={
                     "Tinh": "Tỉnh(Thành phố)",
                     "LoaiBDS": "Loại BDS"
                 },)

    fig1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    #Figure Date
    pd_date = df.copy()
    pd_date['NgayDangBan'] = pd.to_datetime(pd_date['NgayDangBan'], format="%d/%m/%Y, %H:%M").dt.date
    
    fig_date = px.histogram(pd_date, x="NgayDangBan", labels={
                        "NgayDangBan": "Ngày Đăng bán",
                    },)

    fig_dateJSON = json.dumps(fig_date, cls=plotly.utils.PlotlyJSONEncoder)
    
    #Figure 2
    fig2 = px.histogram(df, x="LoaiBDS", y="TongGia", histfunc='avg', labels = {
            "LoaiBDS": "Loại BDS",
            "TongGia": "price"
        })
    fig2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    #Figure 3
    pd_df2 = df.groupby('LoaiBDS').size().reset_index(name='Observation')
    fig3 = px.pie(pd_df2, values='Observation', names='LoaiBDS', title = 'Tỷ lệ các loại BDS')
    fig3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    return {
        "meanPrice" : meanPrice,
        "numOfProject" : numOfProject,
        "fig1JSON" : fig1JSON,
        "fig2JSON" : fig2JSON,
        "fig3JSON" : fig3JSON,
        "fig_dateJSON" : fig_dateJSON,
    }

def read_data(spark):
    df = spark.read.format('org.apache.spark.sql.json').load("./data/clean/clean.json")
    
    data = df.drop(*['id', 'MoTa'])
    data = data.fillna(0)
    pd_df = data.toPandas()

    return pd_df

def init_pre_model():
    string_idx = PipelineModel.load("./model/str_idx")
    enc_m = OneHotEncoderModel.load("./model/ohe_idx")

    return string_idx, enc_m

def init_ml_model():
    model_lr_rmo = LinearRegressionModel.load("./model/linear_regression/lr_outlierRm")
    model_rf_rmo = RandomForestRegressionModel.load("./model/random_forest/rf_outlierRm")
    model_gbt_rmo = GBTRegressionModel.load("./model/gradient_boosted/gbt_outlierRm")
    model_dt_rmo = DecisionTreeRegressionModel.load("./model/decision_tree/dt_outlierRm")
    model_ir_rmo = IsotonicRegressionModel.load("./model/isotonic_regression/ir_outlierRm")

    return [model_lr_rmo, model_rf_rmo, model_gbt_rmo, model_dt_rmo, model_ir_rmo]