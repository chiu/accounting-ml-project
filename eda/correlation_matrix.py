from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext

integrated_df = spark.read.parquet('/user/vcs/annual_integrated_dataset_v2.parquet').limit(1000).cache()
print(integrated_df.head())
df = integrated_df.cache()

# string_columns = [k for (k,v) in integrated_df.dtypes if v == 'string']
# integrated_df = integrated_df.drop(*string_columns)

double_columns = [k for (k,v) in integrated_df.dtypes if v == 'double']
df = integrated_df.select(*double_columns)

print('ultra 6')
#
# import csv
#
# null_count_list = []
# # annual_compustat_null_count.csv
# with open('annual_compustat_null_count.csv', 'r') as f:
#     reader = csv.reader(f)
#     your_list = list(reader)
#     null_count_list = your_list[0]
#     null_count_list = [float(x) for x in null_count_list]


from pyspark.sql.functions import isnan, when, count, col

nullcounts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
null_count_list = list(nullcounts.first())

good_columns = []
total_rows = integrated_df.count()
for i in range(0, len(null_count_list)):
    if null_count_list[i]/total_rows < 0.4:
        good_columns.append(i)

great_columns = [df.columns[i] for i in good_columns]
df = df.select(*great_columns)


from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=df.columns,
    outputCols=["{}_imputed".format(c) for c in df.columns]
)
df = imputer.fit(df).transform(df)

df = df.drop(*great_columns)

df.head()
print('v2 great')


assembler = VectorAssembler(
    inputCols=df.columns,
    outputCol="features")

output = assembler.transform(df)

output.head()

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext

"""
An example for computing correlation matrix.
Run with:
  bin/spark-submit examples/src/main/python/ml/correlation_example.py
"""
# from __future__ import print_function

from pyspark.ml.stat import Correlation
df = output

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))
# $example off$

spark.stop()


