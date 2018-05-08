import csv
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler

feature_set = []
label_set = []

with open('AAPL_data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    for index,line in enumerate(csv_reader):
        if (index > 0):
            feature_set.append(line[1:6])
        
        if (index > 1):
            label_set.append(line[4])
        
    feature_set.pop()

test_feature = []
test_label = []
    
with open ('AAPL.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    for index,line in enumerate(csv_reader):
        if (index > 0):
            test_feature.append(line[1:6])
        
        if (index > 1):
            test_label.append(line[4])
        
    test_feature.pop()
    
for i in range (len(feature_set)):
    label_set[i] = float(label_set[i])
    for a in range (len(feature_set[0])):    
       feature_set[i][a] = float(feature_set[i][a])    
    
for i in range (len(test_feature)):
    test_label[i] = float(test_label[i])
    for a in range (len(test_feature[0])):    
       test_feature[i][a] = float(test_feature[i][a]) 

scale = MinMaxScaler(feature_range=(0,1))

feature_set = np.array(feature_set)
label_set = np.array(label_set)

test_feature = scale.fit_transform(np.array(test_feature))
test_label =scale.fit_transform(np.array(test_label).reshape(-1,1))

feature_set = scale.fit_transform(feature_set)
label_set = scale.fit_transform(label_set.reshape(-1,1))

feature_set = np.reshape(feature_set,(feature_set.shape[0],feature_set.shape[1],1))
test_feature = np.reshape(test_feature,(test_feature.shape[0],test_feature.shape[1],1))


from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout

'''

initial = Input(shape = (5, 1))
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
model.fit(feature_set, label_set, validation_split = 0.1, epochs = 50, verbose = 2)

model.save('Stocks1.h5')

'''

from keras.models import load_model

new_model = load_model('Stocks1.h5')

test_predict = scale.inverse_transform((new_model.predict(test_feature)))

plt.plot(scale.inverse_transform(test_label), color = 'red', label = 'Real AAPL Stock Price')
plt.plot(test_predict, color = 'blue', label = 'Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()
