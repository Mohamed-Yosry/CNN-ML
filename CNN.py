
#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

# loadding and splitting the data
(x_datasetTrain, y_datasetTrain), (x_datasetTest, y_datasetTest) = mnist.load_data()
print(x_datasetTrain.shape)
print(y_datasetTrain.shape)

print(x_datasetTest.shape)
print(y_datasetTest.shape)

# step 1
# normalizing the data and converting to float
x_datasetTrain = x_datasetTrain.astype('float')
x_datasetTest = x_datasetTest.astype('float')
x_datasetTrain = x_datasetTrain/255
x_datasetTest = x_datasetTest/255
x_datasetTrain[0][10]
x_datasetTest[0][10]

x_datasetTrain = x_datasetTrain.reshape(60000,28,28,1)
x_datasetTest = x_datasetTest.reshape(10000,28,28,1)

# make one hot encoding
# machine learning algorithms cannot work with categorical data directly.
# You generate one boolean column for each category or class. 
# Only one of these columns could take on the value 1 for each sample.
y_datasetTrain_encode = to_categorical(y_datasetTrain)
y_datasetTest_endcode = to_categorical(y_datasetTest)
print(y_datasetTest[0])
print(y_datasetTest_endcode[0])

#  step 2
#  make the models
models = [None] * 4

# model index 0
models[0]= Sequential()
# adding convolutional layer with 
models[0].add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
models[0].add(Conv2D(64, kernel_size=3, activation='relu'))
models[0].add(Flatten())
models[0].add(Dense(10, activation='softmax'))

#  model index 1
models[1]= Sequential()
models[1].add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
models[1].add(Conv2D(32, kernel_size=3, activation='relu'))
# add the max-pooling layer 
models[1].add(MaxPooling2D(pool_size=(2,2)))
models[1].add(Conv2D(64, kernel_size=3, activation='relu'))
models[1].add(MaxPooling2D(pool_size=(2,2)))
models[1].add(Conv2D(32, kernel_size=3, activation='relu'))
models[1].add(Flatten())
models[1].add(Dense(10, activation='softmax'))

#  model index 2
models[2]= Sequential()
models[2].add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
models[2].add(Conv2D(32, kernel_size=3, activation='relu'))
models[2].add(MaxPooling2D(pool_size=(2,2)))
models[2].add(Conv2D(64, kernel_size=3, activation='relu'))
models[2].add(Conv2D(32, kernel_size=3, activation='relu'))
models[2].add(MaxPooling2D(pool_size=(2,2)))
models[2].add(Conv2D(64, kernel_size=3, activation='relu'))
models[2].add(MaxPooling2D(pool_size=(2,2)))
models[2].add(Flatten())
models[2].add(Dense(10, activation='softmax'))

#  model index 3
models[3]= Sequential()
models[3].add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
models[3].add(Conv2D(64, kernel_size=3, activation='relu'))
models[3].add(Conv2D(32, kernel_size=3, activation='relu'))
models[3].add(MaxPooling2D(pool_size=(2,2)))
models[3].add(Conv2D(32, kernel_size=3, activation='relu'))
models[3].add(Conv2D(128, kernel_size=3, activation='relu'))
models[3].add(Conv2D(128, kernel_size=3, activation='relu'))
models[3].add(MaxPooling2D(pool_size=(2,2)))
models[3].add(Flatten())
models[3].add(Dense(10, activation='softmax'))

# compiling the models using the Adam optimizer algo
for i in range(len(models)):
  models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  #train[i] = models[i].fit(x_datasetTrain,y_datasetTrain,validation_data=(x_datasetTest,y_datasetTest))

# step 3
# trainning
# applying cross validaion using cross fold
num_of_folds = 4

best_accuracy = 0
best_model = None
best_model_index = 0
stratified_fold = StratifiedKFold(n_splits = num_of_folds, shuffle = True)
train = [None] * 4

for train_index , value_index in stratified_fold.split(x_datasetTrain,y_datasetTrain):
  for i in range(len(models)):
    train[i] = models[i].fit(x_datasetTrain[train_index],y_datasetTrain_encode[train_index],validation_data=(x_datasetTrain[value_index],y_datasetTrain_encode[value_index]))
    accuracy = train[i].history['accuracy'][-1]
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_model = models[i]
      best_model_index = i

# predict the values of x_datasetTest
predicted=best_model.predict(x_datasetTest)
print(predicted)

#prediceted labesls
predicted = np.argmax(predicted,axis=1)
print(predicted)

incorrect_prediction = np.nonzero(predicted != y_datasetTest)[0]
correct_prediciton = np.nonzero(predicted == y_datasetTest)[0]
print("incorrect_prediction:")
print(incorrect_prediction)
print("correct_prediciton")
print(correct_prediciton)
len(incorrect_prediction),len(correct_prediciton)