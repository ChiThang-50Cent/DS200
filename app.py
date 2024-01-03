import streamlit as st
import numpy as np
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

import plotly.express as px

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep

from utils import initialize_spark
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, IsotonicRegressionModel, FMRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoderModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import PipelineModel

from utils import *
from crawl_url import *
from crawl_data import *
from clean_data import *
from train_model import *
from feature_extract import *

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

def tranformFetures(X, use_transform=True):
    string_idx = PipelineModel.load("./model/str_idx")
    enc_m = OneHotEncoderModel.load("./model/ohe_idx")
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
    with st.spinner('Predicting...'):
        # Encode dữ liệu
        X = tranformFetures(samples, use_transform=use_transform)

    pred = model.predict(X.head().features)

    # Lấy kết quả dự đoán.
    # results = pd.DataFrame({'Giá dự đoán': [pred]})

    # Test 
    results = get_result(X.head().TongGia, pred)
    
    # Xuất ra màn hình
    st.write(results)
                            
def load_sample_data(model):
    # Chọn dữ liệu từ mẫu
    selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
    selected_rows = pd_df.iloc[selected_indices]

    st.write('#### Kết quả')
    st.write(selected_rows)

    if st.button('Dự đoán'):
        if not selected_rows.empty:
            
            X = spark.createDataFrame(selected_rows.astype(str))

            prediction(X, model)
        else:
            st.error('Hãy chọn dữ liệu trước')

def inser_data(model):
    with st.form("Nhập dữ liệu"):
        loaiBDS = st.text_input("Loại BDS*", placeholder='Đất bán')
        dienTich = st.text_input("Diện Tích*", placeholder='200')
        tinh = st.text_input("Tỉnh\Thành phố*", placeholder='Bình Thuận')
        hienTrangNha = st.text_input("Hiện Trạng Nhà", placeholder='Nhà trống')
        viTri = st.text_input("Vị trí", placeholder='Mặt tiền')
        phongNgu = st.text_input("Số phòng ngủ", placeholder='2')
        phongTam = st.text_input("Số phòng tắm", placeholder='2')
        tang = st.text_input("Số tầng", placeholder='2')

        submitted = st.form_submit_button("Dự Đoán")

        if submitted:
            data_submitted = {'LoaiBDS' : loaiBDS,
                                'DienTich' : dienTich,
                                'Tinh': tinh,
                                'HienTrangNha': hienTrangNha,
                                'ViTri': viTri,
                                'PhongNgu': phongNgu,
                                'PhongTam': phongTam,
                                'SoTang': tang}
            
            X = pd.DataFrame([data_submitted])
            X = gen_input_data(X, pd_df.iloc[[np.random.randint(700)]].reset_index(drop=True))
            X = spark.createDataFrame(X.astype(str))
            
            prediction(X, model)

def get_data_from_URL(model):
    st.write('#### Crawl dữ liệu từ URL')

    with st.form(key='URL_form'):
        URL = st.text_input(
            label='Điền URL đến bài đăng bán BDS lấy từ https://nhadatvui.vn/ cần dự đoán.\nVd: https://nhadatvui.vn/ban-nha-rieng-phuong-hiep-binh-chanh-tp-thu-duc/ban-tret-lau-hem-ba-gac-duong-49-hiep-binh-chanh-thanh-pho-thu-duc1704279569',
            placeholder='https://nhadatvui.vn/bat-dong-san-ABC')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if not validateURL(URL):
            noti = st.warning('URL không hợp lệ')
        else:
            try:
                with st.spinner('Crawling ...'):
                    status, postInfo = getdata(URL)
            except:
                noti = st.warning("Can't get URL")
            else:
                if status == 200:
                    with st.spinner('Data processing ...'):
                        post_pandasDF = pd.DataFrame([postInfo])
                        post_pandasDF = gen_input_data(post_pandasDF, pd_df.iloc[[np.random.randint(500)]].reset_index(drop=True))
                        
                        st.write(post_pandasDF)

                        post_pDF = spark.createDataFrame(post_pandasDF.astype(str))
                        post_pDF = from_pd_to_spark(post_pDF)
                        post_clean = cleanRawData(post_pDF)

                        output = st.empty()
                        with st_capture(output.code):
                            print(post_clean.show())

                        prediction(post_clean, model, use_transform=False)
                else:
                    print('Cant request url', status)

def model_page(model_name, model):
    option_list = ['Dữ liệu mẫu', 'Nhập dữ liệu', 'Crawl dữ liệu từ URL']
    
    choice_input = st.sidebar.selectbox('Cách nhập dữ liệu', option_list)    
    st.subheader(model_name)

    if choice_input == 'Dữ liệu mẫu':
        st.write('#### Sample dataset', pd_df)
        load_sample_data(model)

    elif choice_input == 'Nhập dữ liệu':
        inser_data(model)

    elif choice_input == 'Crawl dữ liệu từ URL':
        get_data_from_URL(model)

def create_dashboard(df):
    st.subheader('Dashboard')

    col1, col2 = st.columns(2)
    col1.metric(label="Số lượng dự án", value=df.shape[0])
    col2.metric(label="Giá tiền trung bình mỗi dự án",
                value="{:,} VND".format(round(df['TongGia'].mean() * 1000000)))

    fig1 = px.histogram(pd_df, x="Tinh", color="LoaiBDS", labels={
                     "Tinh": "Tỉnh(Thành phố)",
                     "LoaiBDS": "Loại BDS"
                 },)

    pd_date = pd_df.copy()
    pd_date['NgayDangBan'] = pd.to_datetime(pd_date['NgayDangBan'], format="%d/%m/%Y, %H:%M").dt.date
    
    fig_date = px.histogram(pd_date, x="NgayDangBan", labels={
                        "NgayDangBan": "Ngày Đăng bán",
                    },)
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig_date, use_container_width=True)

    fig_col2, fig_col3 = st.columns(2)

    fig2 = px.histogram(pd_df, x="LoaiBDS", y="TongGia", histfunc='avg', labels = {
            "LoaiBDS": "Loại BDS",
            "TongGia": "price"
        })

    pd_df2 = df.groupby('LoaiBDS').size().reset_index(name='Observation')
    fig3 = px.pie(pd_df2, values='Observation', names='LoaiBDS', title = 'Tỷ lệ các loại BDS')

    fig_col2.plotly_chart(fig2)
    fig_col3.plotly_chart(fig3)

    st.write(pd_df)

def main():
    st.title('Dự đoán giá bất động sản')
    model_list = ['Dashboard',
                    'Mô hình Linear Regression',
                    'Mô hình Random Forest',
                    'Mô hình Gradient Boosting',
                    'Mô hình Decision Tree',
                    'Mô hình Isotonic Regression']

    global choice_model
    choice_model = st.sidebar.selectbox('Tùy chọn:', model_list)


    if choice_model =='Dashboard':
        create_dashboard(pd_df)
    elif choice_model == 'Mô hình Linear Regression':
        model_lr_rmo = LinearRegressionModel.load("./model/linear_regression/lr_outlierRm")
        model_page(choice_model, model_lr_rmo)

    elif choice_model == 'Mô hình Random Forest':
        model_rf_rmo = RandomForestRegressionModel.load("./model/random_forest/rf_outlierRm")
        model_page(choice_model, model_rf_rmo)

    elif choice_model == 'Mô hình Gradient Boosting':
        model_gbt_rmo = GBTRegressionModel.load("./model/gradient_boosted/gbt_outlierRm")
        model_page(choice_model, model_gbt_rmo)

    elif choice_model == 'Mô hình Decision Tree':
        model_dt_rmo = DecisionTreeRegressionModel.load("./model/decision_tree/dt_outlierRm")
        model_page(choice_model, model_dt_rmo)

    elif choice_model == 'Mô hình Isotonic Regression':
        model_ir_rmo = IsotonicRegressionModel.load("./model/isotonic_regression/ir_outlierRm")
        model_page(choice_model, model_ir_rmo)


if __name__ == '__main__':
    spark, sc = initialize_spark()
    st.set_page_config(layout="wide")

    ## Load dataset
    with st.spinner('Load data...'):
        df = spark.read.format('org.apache.spark.sql.json').load("./data/clean/clean.json")
    
    data = df.drop(*['id', 'MoTa'])
    data = data.fillna(0)
    pd_df = data.toPandas()

    output = st.empty()
        
    main()
