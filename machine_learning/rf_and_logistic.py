import os

os.environ['SPARK_HOME'] = '/home/envmodules/lib/spark-2.2.0-bin-hadoop2.7/'
import findspark

findspark.init()

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

import pandas as pd

spark = SparkSession.builder.appName('733').getOrCreate()

# Using the integrated file to start working on
integrated_df = spark.read.parquet('/user/vcs/annual_integrated_dataset_with_labels_ibes_fix_v2.parquet').cache()


def find_performance_metrics(res, model_used):
    res = res.withColumn('correct', res.label == res.prediction)

    num_rows = res.count()
    accuracy = res.filter(res.label == res.prediction).count() / res.count()

    # positive class (misstatements)
    true_positives_df = res.filter(res.prediction == 1.0).filter(res.label == 1.0)
    ground_truth_positives_df = res.filter(res.label == 1.0)
    misstatement_recall = true_positives_df.count() / ground_truth_positives_df.count()

    new_all_predicted_positive_df = res.filter(res.prediction == 1.0)
    misstatement_precision = true_positives_df.count() / new_all_predicted_positive_df.count()

    # negative class (non misstatements)
    true_negative_df = res.filter(res.prediction == 0.0).filter(res.label == 0.0)
    ground_truth_negative_df = res.filter(res.label == 0.0)
    non_misstatement_recall = true_negative_df.count() / ground_truth_negative_df.count()

    new_all_predicted_negative_df = res.filter(res.prediction == 0.0)
    non_misstatement_precision = true_negative_df.count() / new_all_predicted_negative_df.count()

    d = {'model_used': model_used, 'accuracy': accuracy, \
         'misstatement_precision': misstatement_precision, \
         'misstatement_recall': misstatement_recall}
    df = pd.DataFrame(data=d, index=[0])
    file_name = "performance_metrics" + "".join(model_used.split()) + ".csv"
    df.to_csv(file_name, encoding='utf-8')

    print("Using {}".format(model_used))
    print('accuracy is {}'.format(accuracy))
    print('misstatement_precision is {}, misstatement recall is {}'.format(misstatement_precision, misstatement_recall))
    print('non_misstatement_precision is {}, non_misstatement recall is {}'.format(non_misstatement_precision,
                                                                                   non_misstatement_recall))


# Downsampling:
misstated_df = integrated_df.filter(integrated_df.label == 1.0)
misstated_count = misstated_df.count()
non_misstated_df = integrated_df.filter(integrated_df.label == 0.0).limit(misstated_count)
integrated_df = misstated_df.union(non_misstated_df).cache()

# Using nullcounts to filter columns to keep
nullcounts = integrated_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in integrated_df.columns])
nc = list(nullcounts.first())

# Services-packaged software category selection (from EDA)
services_prepacked_software = integrated_df  # .filter(integrated_df.sic == '7372')
print('Total records in integrated file: ', integrated_df.count())
print('Number of records in Services-packaged software industrial category: ', services_prepacked_software.count())

# Reusing preprocessing steps implemented by Vincent
# filling nulls and nones with zeroes.
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

# dropping all string columns
non_string_columns = [k for (k, v) in nwdf.dtypes if v != 'string']
nwdf_no_strings = nwdf.select(*non_string_columns)
feature_columns = [item for item in nwdf_no_strings.columns if item not in ['rea', 'features', 'label', 'rea_label']]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_df = assembler.transform(nwdf_no_strings)
final_final_df = final_df.drop(*feature_columns).cache()

# String indexing not required
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(final_final_df)
td = si_model.transform(final_final_df)

# Evaluators
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
eval = BinaryClassificationEvaluator()

# RandomForest classifier
rf = RandomForestClassifier(numTrees=100, maxDepth=16, labelCol="indexed", seed=42)
model = rf.fit(td)
result = model.transform(final_final_df)
print('Accuracy on training data: ', evaluator.evaluate(result))

# Train test split for model evaluation
train, test = final_final_df.randomSplit([0.7, 0.3], seed=12345)
train.cache()
test.cache()

# ---------------
# Random Forest:
# ---------------

rf = RandomForestClassifier(numTrees=100, maxDepth=16, labelCol="label", seed=42)
print('Training RandomForest model on training set. \n Model parameters: {}'.format(rf._paramMap))
trained_model = rf.fit(train)
res = trained_model.transform(test)
metrics = MulticlassMetrics(res.select(['label', 'prediction']).rdd)
print('Accuracy on test set: ', evaluator.evaluate(res))
print('Area under ROC curve: ', eval.evaluate(res))
find_performance_metrics(res, "random forest")

# ---------------
# Logistic regression:
# ---------------
print('Training LogisticRegression model on training set.')
logistic = LogisticRegression(regParam=0.1, labelCol="label")  # , thresholds = [0.2, 0.5])
trained_model = logistic.fit(train)
res = trained_model.transform(test)
metrics = MulticlassMetrics(res.select(['label', 'prediction']).rdd)
print('Accuracy on test set: ', evaluator.evaluate(res))
print('Area under ROC curve: ', eval.evaluate(res))
find_performance_metrics(res, "logistic regression")

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = trained_model.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']).select('threshold').head()[
    'threshold']
logistic.setThreshold(bestThreshold)
print('best threshold is:' + str(bestThreshold))

print("For Logistic regression:")
trained_model = logistic.fit(train)
res = trained_model.transform(test)
metrics = MulticlassMetrics(res.select(['label', 'prediction']).rdd)
print('Accuracy on test set: ', evaluator.evaluate(res))
print('Area under ROC curve: ', eval.evaluate(res))
# find_performance_metrics(res, "logistic regression")
find_performance_metrics(res, "logistic regression with best threshold")

df = pd.DataFrame(
    {'lr_coeff': trained_model.coefficients,
     'feature_column': feature_columns,
     })

df['abs_lr_coeff'] = df['lr_coeff'].abs()
df = df = df.sort_values('abs_lr_coeff', ascending=False).reset_index()
print(df.head())

# ------------------------------------------------------------
# Code for making use of validation set for parameter tuning
train, test = final_final_df.randomSplit([0.9, 0.1], seed=12345)

lr = LogisticRegression()

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.2, 0.15, 0.1, 0.01]) \
    .addGrid(lr.threshold, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])\
    .build()

# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
res = model.transform(test)
find_performance_metrics(res, 'logistic_with_validation')

# ---------------------------------------------------------
# For RandomForest
rf = RandomForestClassifier()

paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [50, 100, 150, 200]) \
    .addGrid(rf.maxDepth, [4, 8, 12, 16, 18, 20])\
    .build()
    
tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)
model = tvs.fit(train)
res = model.transform(test)
find_performance_metrics(res, 'rf_with_validation')
