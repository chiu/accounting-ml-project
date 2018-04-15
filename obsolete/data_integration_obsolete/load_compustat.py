from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext

df = spark.read.csv('./quarterly reports.csv', header=True)

# Number of total records
df.count()

nullcounts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
nullcounts.write.csv('nullcount')

