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


ml_df.show()
# ml_df = ml_df.withColumn('boolean_label', ml_df.rea != 0)
# ml_df = ml_df.withColumn('label', ml_df.boolean_label.cast('float'))
#
# ml_df.show()
#
#
#
#
# ml_df = ml_df.drop('rea').drop('boolean_label')
#
#
#
#
# # Split the data into train and test
# splits = ml_df.randomSplit([0.6, 0.4], 12)
# train = splits[0]
# test = splits[1]
#
# # specify layers for the neural network:
# # input layer of size 4 (features), two intermediate of size 5 and 4
# # and output of size 3 (classes)
# layers = [1514, 1514, 1514, 2]
#
# # create the trainer and set its parameters
# trainer = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)
#
#
# # In[32]:
#
#
# # train the model
# # model = trainer.fit(train)
#
#
# # In[33]:
#
#
# # # compute accuracy on the test set
# # result = model.transform(test)
# # predictionAndLabels = result.select("prediction", "label")
# # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
# # print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
#
#
# # In[34]:
#
#
# import numpy as np
# label_np = np.array(train.select('label').collect())
#
#
# # In[35]:
#
#
# features_np = np.array(train.select('features').collect())
#
#
# # In[36]:
#
#
# features_np_flat = [x[0] for x in features_np]
#
#
# # In[37]:
#
#
# result = np.vstack(features_np_flat)
#
#
# # In[38]:
#
#
# # For a single-input model with 2 classes (binary classification):
# from keras.models import Model
# from keras.layers import Input, Dense
# from keras.models import Sequential
# model = Sequential()
# model.add(Dense(18, activation='relu', input_dim=9))
# # model.add(Dense(18, activation='relu', input_dim=18))
# model.add(Dense(1, input_dim = 18,  activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Generate dummy data
# # import numpy as np
# # data = np.random.random((1000, 100))
# # labels = np.random.randint(2, size=(1000, 1))
#
# # Train the model, iterating on the data in batches of 32 samples
# model.fit(result, label_np, epochs=100, batch_size=32)
#
#
# # In[39]:
#
#
# unique, counts = np.unique(label_np, return_counts=True)
#
#
# # In[40]:
#
#
# unique
#
#
# # In[41]:
#
#
# counts
#
#
# # In[42]:
#
#
# 547/(49+547)
#
#
#
#
#
