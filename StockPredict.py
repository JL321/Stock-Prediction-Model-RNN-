import csv
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

train_df = pd.read_csv('AAPL_data.csv', index_col = 0)
relevantData = np.array(train_df[['open','high','low','close','volume']])
relevantData = relevantData.tolist()


dataBatch = []
inputSet = []
labelSet = []

flag = True

counter = 1

#Inputs are consisted of 5 inputs over 5 days - closing, low, opening, high, and volume

while flag == True:
    inputSet.append(relevantData[counter][:])
    inputSet.append(relevantData[counter+1][:])
    inputSet.append(relevantData[counter+2][:])
    inputSet.append(relevantData[counter+3][:])
    inputSet.append(relevantData[counter+4][:])
    
    inputSet = np.array(inputSet)
    inputSet = inputSet.flatten()
    inputSet = inputSet.tolist()
    dataBatch.append(inputSet)
    inputSet = []
    labelSet.append(relevantData[counter+5][3]) #append the closing price - this will be the label
    
    counter += 1
    if ((len(relevantData))-counter < 6):
        flag = False
        
test_df = pd.read_csv('AAPL.csv', index_col = 0)
relevantDataT = np.array(test_df[['Open','High','Low','Close','Volume']])
relevantDataT = relevantDataT.tolist()
counter = 1

dataBatchT = []
inputTest = []
labelTest = []

flag = True

while flag == True: 
    inputTest.append(relevantDataT[counter][:])
    inputTest.append(relevantDataT[counter+1][:])
    inputTest.append(relevantDataT[counter+2][:])
    inputTest.append(relevantDataT[counter+3][:])
    inputTest.append(relevantDataT[counter+4][:])
    inputTest = np.array(inputTest)
    inputTest = inputTest.flatten()
    inputTest = inputTest.tolist()
    dataBatchT.append(inputTest)
    inputTest = []
    labelTest.append(relevantDataT[counter+5][3]) #append the closing price - this will be the label
    counter += 1
    if ((len(relevantDataT))-counter < 6):
        flag = False

label_set = np.array(labelSet)
test_label = np.array(labelTest)
test_feature = np.array(dataBatchT)
feature_set = np.array(dataBatch)

#print(dataBatch)

print (feature_set.shape)
print (label_set.shape)

scale = MinMaxScaler(feature_range=(0,1))

feature_set = scale.fit_transform(feature_set)
label_set = scale.fit_transform(label_set.reshape(-1,1))

test_feature = scale.fit_transform(test_feature)
test_label =scale.fit_transform(test_label.reshape(-1,1))

feature_set = np.reshape(feature_set,(feature_set.shape[0],feature_set.shape[1],1))
test_feature = np.reshape(test_feature,(test_feature.shape[0],test_feature.shape[1],1))

print(feature_set.shape)
print(test_feature.shape)

from keras.models import Model #Start building model
from keras.layers import Input, LSTM, Dense, Dropout

initial = Input(shape = (25, 1))
x = LSTM(128, return_sequences = True)(initial)
x = Dropout(0.8)(x)
x = LSTM(128, return_sequences = True)(x)
x = Dropout(0.8)(x)
x = LSTM(128, return_sequences = True)(x)
x = Dropout(0.8)(x)
x = LSTM(128, return_sequences = False)(x)
x = Dense(1)(x)

model = Model(inputs = initial, outputs = x)

model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
model.fit(feature_set, label_set, validation_split = 0.1, epochs = 30, verbose = 2)

model.save('Stocks25.h5')


from keras.models import load_model


new_model = load_model('Stocks25.h5')

test_predict = scale.inverse_transform((new_model.predict(feature_set)))

plt.plot(scale.inverse_transform(label_set), color = 'red', label = 'Real AAPL Stock Price')
plt.plot(test_predict, color = 'blue', label = 'Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()
