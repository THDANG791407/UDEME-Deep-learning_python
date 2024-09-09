#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


data = pd.read_csv('Google_Stock_Price_Train.csv')
data


# In[3]:


train_set = data.iloc[:,1:2].values
train_set


# In[4]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # pp chuẩn hóa dữ liệu biến dữ liệu về giá trị 0 , 1
training_set_scaled = sc.fit_transform(train_set) # fit chuẩn hóa , transform chuẩn tất cả giá trị thành chuẩn hóa
training_set_scaled


# Tạo cấu trúc dữ liệu 60 bước thời gian và 1 đầu ra

# In[5]:


X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[6]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# chuyển dữ liệu X_train thành mảng 1 chiều


# Bước 2: xây dựng RNN

# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import LSTM, Dropout
from tensorflow.keras import layers


# In[8]:


# khởi tạo mạng
regressor = Sequential()


# In[9]:


#Thêm lớp LSTM đầu tiên và một số quy tắc Dropout
regressor.add(layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # return_sequences = True trả về toàn bộ chuổi đầu ra ở mỗi bước thời gian cho các đơn vị ận thây vì trả về chuỗi đầu ra cuối cùng
regressor.add(layers.Dropout(0.2)) # Dropout tránh mô hình bị overfiting 20% neuron lớp trước đó sẽ bị vô hiệu hóa sau mỗi lần huấn luyện


# In[10]:


print(tf.__version__)


# In[11]:


print(tf.keras.__version__)


# In[12]:


#Thêm lớp LSTM thứ 2 và một số quy tắc Dropout
regressor.add(layers.LSTM(units = 50, return_sequences = True))
regressor.add(layers.Dropout(0.2))


# In[13]:


#Thêm lớp LSTM thứ 3 và một số quy tắc Dropout
regressor.add(layers.LSTM(units = 50, return_sequences = True))
regressor.add(layers.Dropout(0.2))


# In[14]:


#Thêm lớp LSTM thứ 4 và một số quy tắc Dropout
regressor.add(layers.LSTM(units = 50))
regressor.add(layers.Dropout(0.2))


# In[15]:


# thêm đầu ra
regressor.add(Dense(units = 1))


# In[16]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[17]:


regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[18]:


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# In[27]:


dataset_total = pd.concat((data['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1) # chuyển giá trị về mảng 2D
inputs = sc.transform(inputs) # chuẩn hóa dữ liệu
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) # i-60:i nghĩa là lấy giá trị từ 0-59 dự đoán giá ngày 60, còn số 0 nghĩa là lấy cột 0 inputs chỉ có 1 cột nên là số 0)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test) # giá trị dự đoán đang ở dạng 0 đến 1
print(predicted_stock_price)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # chuyển giá trị dự đoán về dạng giá gốc của cổ phiếu 


# In[28]:


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[29]:


len(dataset_total)


# In[30]:


print(predicted_stock_price)


# In[23]:


print(real_stock_price)


# In[26]:


print(inputs)


# In[ ]:




