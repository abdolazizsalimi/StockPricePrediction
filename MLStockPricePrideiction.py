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


# ------------------------------------constans-----------------------------------------------------
start_date = '2020-01-01'
end_date   = '2021-05-30'
DataSetFor = 'AAPL'
# =================================================================================================
# Get  the stock qoute  
df = web.DataReader( DataSetFor, data_source = 'yahoo' , start = start_date , end = end_date)
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
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1] ,1 ))

#  Bulid the LSTM model 
model = Sequential()
model.add(LSTM(50,return_sequences=True , input_shape= (x_train.shape[1],1))) 
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))  
model.add(Dense(1))  


# compile the model 
model.compile(optimizer='adam',loss='mean_squared_error' ) 
# traine the model 
model.fit(x_train , y_train ,  batch_size=1 , epochs= 1 )

# Creat the testing data set 
# Creat a new array containing scaled values from index 1543 to 2003 
test_data = scaled_data[training_data_len-60 : , : ]
# creat the data sets x_test , y_test 
x_test = [] 
y_test = dataset[training_data_len : , :]
for i in range(60 , len(test_data)): 
    x_test.append(test_data[i-60:i, 0])  


# Convert the data to a numpy array 
x_test =np.array(x_test)
# reshape the data 
x_test = np.reshape(x_test , (x_test.shape[0] , x_test.shape[1] ,1 ))  

# Get the model predictions price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test.shape , predictions.shape


# Get the root mean Squared error (RMSE)
rmes = np.sqrt( np.mean(predictions-y_test )**2) 



# plot the data 
train = data[:training_data_len]
valid = data[training_data_len:] 
valid['Predictions'] = predictions
# Visualiize the dasta 
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date' , fontsize = 18 )
plt.ylabel('Close price USD($)' , fontsize = 18 )
plt.plot(train["Close"])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
plt.show() 

