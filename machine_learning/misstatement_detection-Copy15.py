
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
spark = SparkSession.builder.appName('733').getOrCreate()
sc = spark.sparkContext


# In[2]:


annual_df = spark.read.csv('../annual_compustat.csv', header=True, inferSchema=True).limit(1000).cache()


# In[3]:


nullcounts = spark.read.csv('annual_compustat_null_count.csv', header=False)


# In[4]:


import csv

with open('annual_compustat_null_count.csv', 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)



# In[5]:


null_count_list = your_list[0]


# In[6]:


null_count_list = [float(x) for x in null_count_list]


# In[7]:


good_columns = []
for i in range(0, len(null_count_list)):
    if null_count_list[i]==0:
        good_columns.append(i)


# In[8]:


great_columns = [annual_df.columns[i] for i in good_columns]


# In[9]:


great_columns.append('rea')


# In[10]:


columns_num = [3, 10, 14]
annual_df = annual_df.select(*great_columns)


# In[11]:


some_dict = {}
for x in annual_df.columns:
    some_dict[x] = 0


# In[12]:


permuted_annual_df = annual_df.fillna(some_dict)


# In[13]:


permuted_annual_dtypes = permuted_annual_df.dtypes


# In[14]:


non_string_columns = [k for (k,v) in permuted_annual_dtypes if v != 'string']


# In[15]:


permuted_annual_df_no_strings = permuted_annual_df.select(*non_string_columns)


# In[16]:


feature_columns = [item for item in permuted_annual_df_no_strings.columns if item not in ['rea', 'features']]


# In[17]:


from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=feature_columns, outputCol="features")

final_df = assembler.transform(permuted_annual_df_no_strings
)


# In[18]:


final_final_df = final_df.drop(*feature_columns)


# In[19]:


final_final_df.show()


# In[20]:


final_final_df = final_final_df.withColumn('label', final_final_df.rea)


# In[21]:


final_final_df.show()


# In[22]:


# final_final_df.write.parquet("final_final_df2.parquet")


# In[23]:


from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
ml_df = sqlContext.read.parquet("final_final_df2.parquet")


# In[24]:


ml_df.show()


# In[25]:


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
train = final_final_df
lrModel = lr.fit(train)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[26]:


ml_df.show()


# In[27]:


ml_df = ml_df.withColumn('boolean_label', ml_df.rea != 0)


# In[28]:


ml_df = ml_df.withColumn('label', ml_df.boolean_label.cast('float'))


# In[29]:


ml_df.show()


# In[30]:


ml_df = ml_df.drop('rea').drop('boolean_label')


# In[31]:


# Split the data into train and test
splits = ml_df.randomSplit([0.6, 0.4], 12)
train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [1514, 1514, 1514, 2]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)


# In[32]:


# train the model
# model = trainer.fit(train)


# In[33]:


# # compute accuracy on the test set
# result = model.transform(test)
# predictionAndLabels = result.select("prediction", "label")
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
# print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


# In[34]:


import numpy as np
label_np = np.array(train.select('label').collect())


# In[35]:


features_np = np.array(train.select('features').collect())


# In[36]:


features_np_flat = [x[0] for x in features_np]


# In[37]:


result = np.vstack(features_np_flat)


# In[38]:


# For a single-input model with 2 classes (binary classification):
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import Sequential
model = Sequential()
model.add(Dense(18, activation='relu', input_dim=9))
# model.add(Dense(18, activation='relu', input_dim=18))
model.add(Dense(1, input_dim = 18,  activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(result, label_np, epochs=100, batch_size=32)


# In[39]:


unique, counts = np.unique(label_np, return_counts=True)


# In[40]:


unique


# In[41]:


counts


# In[42]:


547/(49+547)

