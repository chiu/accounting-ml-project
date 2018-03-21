# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('733').getOrCreate()

# Using the integrated file to start working on
integrated_df = spark.read.parquet('/user/vcs/annual_integrated_dataset_v2.parquet')

# Using nullcounts to filter columns to keep
nullcounts = integrated_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in integrated_df.columns])
nc = list(nullcounts.first())

# Extracting out an industrial segment and modelling on it instead of the whole dataset
# Services-packaged software category selection (from EDA)
services_prepacked_software = integrated_df.filter(integrated_df.sic == '7372')
print('Total records in integrated file: ', integrated_df.count())
print(services_prepacked_software.show())
print('Number of records in Services-packaged software industrial category: ', services_prepacked_software.count())

# Reusing preprocessing steps implemented by Vincent
some_dict = {}
for x in services_prepacked_software.columns:
	some_dict[x] = 0

nwdf = services_prepacked_software.fillna(some_dict)

good_columns = []
for i in range(0, len(nc)):
	if nc[i] == 0:
		good_columns.append(i)

great_columns = [nwdf.columns[i] for i in good_columns]
great_columns.append('rea')
nwdf = nwdf.fillna(some_dict)

non_string_columns = [k for (k,v) in nwdf.dtypes if v != 'string']
nwdf_no_strings = nwdf.select(*non_string_columns)
feature_columns = [item for item in nwdf_no_strings.columns if item not in ['rea', 'features']]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_df = assembler.transform(nwdf_no_strings)
final_final_df = final_df.drop(*feature_columns)

final_final_df.withColumn('boolean_label', final_final_df.rea != 0)
final_final_df = final_final_df.withColumn('boolean_label', final_final_df.rea != 0)

print('Class distribution: ', final_final_df.groupBy('boolean_label').count().show())

final_final_df = final_final_df.withColumn('label', final_final_df.boolean_label.cast('float'))
final_final_df = final_final_df.drop('rea').drop('boolean_label')
print(final_final_df.show())

stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(final_final_df)
td = si_model.transform(final_final_df)

# Binary class classification thus using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()

# RandomForest classifier
rf = RandomForestClassifier(numTrees=100, maxDepth=16, labelCol="indexed", seed=42)
model = rf.fit(td)
result = model.transform(final_final_df)
print('Accuracy on training data: ', evaluator.evaluate(result))

# Train test split for model evaluation
train, test = final_final_df.randomSplit([0.7, 0.3], seed=12345)

rf = RandomForestClassifier(numTrees=100, maxDepth=16, labelCol="label", seed=42)
trained_model = rf.fit(train)
res = trained_model.transform(test)
print('Accuracy on test set: ', evaluator.evaluate(res))

# Logistic regression
logistic = LogisticRegression(regParam=0.1, labelCol="label")
trained_model = logistic.fit(train)
res = trained_model.transform(test)
print('Accuracy on test set: ', evaluator.evaluate(res))

