from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext

integrated_df = spark.read.parquet('/user/vcs/annual_integrated_dataset_v2.parquet').limit(1000).cache()
print(integrated_df.head())
print('v4 encoded')
df = integrated_df.cache()

string_columns = [k for (k,v) in integrated_df.dtypes if v == 'string']
integrated_df = integrated_df.drop(*string_columns)
ml_df = integrated_df
ml_df = ml_df.withColumn('boolean_label', ml_df.rea != 0)
ml_df.show()
