from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext
#
# df = spark.read.csv('/user/vcs/annual_integrated_dataset_csv', header=False, inferSchema=True)
#
#
# categorical_cols = ['_c3']
#
#
#
# stringIndexer = StringIndexer(inputCol="_c3", outputCol="categoryIndex")
# model = stringIndexer.fit(df)
# indexed = model.transform(df)
#
# encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
# encoded = encoder.transform(indexed)
# encoded.head()



integrated_df = spark.read.parquet('/user/vcs/annual_integrated_dataset_v2.parquet').limit(1000).cache()
print(integrated_df.head())
print('v4 encoded')
df = integrated_df.cache()


string_columns = [k for (k,v) in integrated_df.dtypes if v == 'string']
integrated_df = integrated_df.drop(*string_columns)
ml_df = integrated_df
ml_df = ml_df.withColumn('boolean_label', ml_df.rea != 0)






# string_column = 'BRIBES'
#
# stringIndexer = StringIndexer(inputCol=string_column, outputCol="categoryIndex")
# model = stringIndexer.fit(df)
# indexed = model.transform(df)
#
# encoder = OneHotEncoder(inputCol="categoryIndex", outputCol= string_column+"Vec")
# encoded = encoder.transform(indexed)
# df = encoded
# df.show()


# permuted_annual_dtypes = integrated_df.dtypes
#
#
# # In[14]:
#
#
# string_columns = [k for (k,v) in permuted_annual_dtypes if v == 'string']
# df = integrated_df.cache()
#
# for string_column in string_columns:
#     try:
#         stringIndexer = StringIndexer(inputCol=string_column, outputCol="categoryIndex")
#         model = stringIndexer.fit(df)
#         indexed = model.transform(df)
#
#         encoder = OneHotEncoder(inputCol="categoryIndex", outputCol= string_column+"Vec")
#         encoded = encoder.transform(indexed)
#         df = encoded
#
#     except:
#         print(string_column)
#
# df.head()
#
# try:
#     df.write.parquet('annual_integrated_dataset_v2_1hot.parquet')
# except:
#     pass

