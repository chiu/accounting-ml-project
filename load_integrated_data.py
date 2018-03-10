from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext

df = spark.read.csv('/user/vcs/annual_integrated_dataset_csv', header=False, inferSchema=True)


categorical_cols = ['_c3']



stringIndexer = StringIndexer(inputCol="_c3", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.head()




print('v3 encoded')