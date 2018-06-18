from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
import csv
from tqdm import tqdm
from math import sqrt

import matplotlib.pyplot as plt


np.random.seed(1671)  # for reproducibility

# network and training
NB_EPOCH = 50
BATCH_SIZE = 256
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
OPTIMIZER = 'Adadelta' # optimizer, explainedin this chapter
N_HIDDEN = 256
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

# data: shuffled and split between train and test sets
name="winequality-red.csv"
data=open(name, "r")
reader= csv.reader(data)
xList = []
labels = []
names = []
firstLine = True

for line in tqdm(data):
	if firstLine:
		names = line.strip().split(";")
		firstLine = False
	else:
		#split on semi-colon
		row = line.strip().split(";")
		#put labels in separate array
		labels.append(float(row[-1]))
		#remove label from row
		row.pop()
		#convert row to floats
		floatRow = [float(num) for num in row]
		xList.append(floatRow)


#Normalize columns in x and labels
nrows = len(xList)
ncols = len(xList[0])
print(nrows, ncols)
#calculate means and variances
xMeans = []
xSD = []
for i in tqdm(range(ncols)):
	col = [xList[j][i] for j in range(nrows)]
	mean = sum(col)/nrows
	xMeans.append(mean)
	colDiff = [(xList[j][i] - mean) for j in range(nrows)]
	sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
	stdDev = sqrt(sumSq/nrows)
	xSD.append(stdDev)

#use calculate mean and standard deviation to normalize xList
xNormalized = []
for i in range(nrows):
	rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
	xNormalized.append(rowNormalized)
#Normalize labels
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)
labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]

#divide attributes and labels into training and test sets
indices = range(len(xList))
X_test = [xNormalized[i] for i in indices if i%7 == 0 ]
X_train = [xNormalized[i] for i in indices if i%7 != 0 ]
Y_test = [labelNormalized[i] for i in indices if i%7 == 0]
Y_train = [labelNormalized[i] for i in indices if i%7 != 0]

print(len(X_train), len(X_train[1]))
print(len(X_test), len(X_test[1]))

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)


# x = list(reader)
# data = numpy.array(x).astype('float')
RESHAPED = 11
X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
Y_train = np.array(Y_train).astype('float32')
Y_test = np.array(Y_test).astype('float32')

X_train = X_train.reshape(1370, RESHAPED)
X_test = X_test.reshape(229, RESHAPED)


# M_HIDDEN hidden layers
# 10 outputs
# final stage is softmax

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
