# Ddescription : this program uses an artificial recurrent neural network called long short term memory (LSTM) 
#                to predict the closing stock price of a corporation (Apple Inc ) using the past 60 day stock price  



#import the laibraries 
import math 
import pandas_datareader as web
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import  MinMaxScaler 
from keras.models import Sequential 
from keras.layers import Dense , LSTM 
import matplotlib.pyplot as plt 

# Get  the stock qoute  
df = web.DataReader('AAPL', data_source = 'yahoo' , start = '2020-01-01', end = '2021-05-30')
# show ten data 
datainfo = pd.DataFrame(df)
datainfo.info()  

# Visualize the closing price history 
plt.figure(figsize=(16,8))
plt.title('Close price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Close Price USD ($)', fontsize= 18)
plt.show() 

# creat a new dataframe with only the close colum
data = df.filter(['Close']) 
#Convert the dataframe to a numpy arrry 
dataset = data.values
# Get the number of rows to train the model 
training_data_len = math.ceil( len(dataset) * .8 ) 
training_data_len 

# Scale the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#Creat the triaining data set 
#Creat the scaled training data set 
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets 
x_train = []
y_train = []

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i , 0])
  y_train.append(train_data[i,0])
  if i<=61 :
    print('x_train: ')
    print(x_train)
    print('------------------------------------------------------------------------')
    print('y_train: ')
    print(y_train)
    print() 


#convert the x_trian,y_trian to numpy arrys 
x_train,y_train = np.array(x_train), np.array(y_train)
# reshape the data 
x_train.shape 


#  Bulid the LSTM model 
model = Sequential()
model.add(LSTM(50,return_sequences=True , input_shape= (x_train.shape[1],1))) 
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))  
model.add(Dense(1))  


# compile the model 
model.compile(optimizer='adam',loss='mean_squared_error' ) 


