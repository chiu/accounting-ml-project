# import numpy as np
# import keras
#
# label_np = np.array(train.select('label').collect())
# features_np = np.array(train.select('features').collect())
# features_np_flat = [x[0] for x in features_np]
# result = np.vstack(features_np_flat)
#
# # For a single-input model with 2 classes (binary classification):
# from keras.models import Model
# from keras.layers import Input, Dense
# from keras.models import Sequential
# model = Sequential()
# model.add(Dense(18, activation='relu', input_dim=9))
# model.add(Dense(1, input_dim = 18,  activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model, iterating on the data in batches of 32 samples
# model.fit(result, label_np, epochs=100, batch_size=32)
# unique, counts = np.unique(label_np, return_counts=True)