from utils import *
from feature_extract import *

spark, sc = initialize_spark()

df = spark.read.format('org.apache.spark.sql.json')\
                .load("./data/clean/clean.json")

df_t = df.limit(5)

X = featureExtraction(df_t)
print(X.dtypes)