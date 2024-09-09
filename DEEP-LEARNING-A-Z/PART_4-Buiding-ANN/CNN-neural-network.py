#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


print(tf.__version__)


# Preprocessing the training set

# In[7]:


train_datagen = ImageDataGenerator(rescale = 1./255, # chia mỗi pixel cho 255 chuyển về giá trị [0,1] thay vì [0,255]
                                   shear_range = 0.2, # làm biến dạng ảnh theo chiều dọc và ngang
                                   zoom_range = 0.2, # phóng to ngẫu nhiên 20% 
                                   horizontal_flip = True) # cho phép lật ngẫu nhiên theo chiều ngang 
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64,64), # mỗi ảnh sẽ được điều chỉnh về 64x64 pixel dù ảnh có lớn
                                                 batch_size = 32, #Tập dữ liệu sẽ được chia thành các batch với kích thước 32 hình ảnh mỗi batch, giúp xử lý dữ liệu theo từng lô thay vì toàn bộ tập cùng một lúc.
                                                 class_mode = 'binary') # chế độ phân loại nhị phân bài này phù hợp vói phân loại chó và mèo


# Preprocessing the test set

# In[10]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64,64), 
                                                 batch_size = 32, 
                                                 class_mode = 'binary')


# Khởi tạo cnn

# In[11]:


cnn = tf.keras.models.Sequential()


# Bước 1: Tích chập (Convolution)

# In[14]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size = 3, activation='relu', input_shape=[64,64,3]))
#filters=32: Số lượng bộ lọc (filters) trong lớp tích chập là 32. Mỗi bộ lọc sẽ học cách phát hiện các đặc trưng khác nhau của hình ảnh (ví dụ: các cạnh, góc, màu sắc).
#kernel_size=3: Kích thước của từng bộ lọc là 3x3. Điều này có nghĩa là mỗi lần áp dụng tích chập (convolution), một lưới 3x3 pixel của hình ảnh sẽ được quét qua.
#activation='relu': Hàm kích hoạt ReLU (Rectified Linear Unit) được áp dụng sau mỗi bước tích chập. ReLU giúp loại bỏ các giá trị âm, chỉ giữ lại các giá trị dương, giúp mạng học tốt hơn và giảm thiểu hiện tượng vanishing gradient.
#input_shape=[64,64,3]: Đây là kích thước đầu vào của ảnh. Ảnh đầu vào có kích thước 64x64 pixel và có 3 kênh màu (RGB). Lớp đầu tiên cần biết kích thước đầu vào này.


# Bước 2: Tập hợp (Pooling)

# In[15]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#pool_size=2: Kích thước của vùng pooling là 2x2. Lớp này sẽ chọn giá trị lớn nhất trong mỗi vùng 2x2 của ảnh để giữ lại.
#strides=2: Bước di chuyển (stride) của vùng pooling là 2 pixel, nghĩa là nó sẽ lấy mẫu giảm kích thước của ảnh đầu vào bằng cách chia kích thước của ảnh xuống một nửa. Điều này giúp giảm số lượng tham số và giảm độ phức tạp của mạng.


# In[16]:


#Thêm lớp chập thứ 2
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size = 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# Bước 3: Làm phẳng (Flattening)

# In[20]:


cnn.add(tf.keras.layers.Flatten())
#Flatten(): Lớp này "trải phẳng" đầu ra của các lớp trước (một tensor 2D hoặc 3D) thành một vector 1D. Nó chuẩn bị dữ liệu để có thể đưa vào các lớp fully connected phía sau (các lớp Dense) cho mục đích phân loại.
#Ví dụ: Nếu đầu ra của lớp MaxPooling có kích thước (16, 16, 32), thì sau khi áp dụng Flatten(), đầu ra sẽ trở thành một vector có kích thước (16 * 16 * 32) = 8192 giá trị.


# Bước 4: kết nối đầy đủ

# In[21]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Lớp 128 noron mỗi nr tính toán dựa trên tất cả nr của lớp trước
#lớp dense sử dụng đặc trưng đã được trích xuất từ các lớp trước để dự đoán


# Bước 5: Lớp đầu ra

# In[22]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Huấn luyện CNN

# In[23]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[24]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[26]:


import numpy as np
from tensorflow.keras.preprocessing import image


# In[44]:


test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) #chuyển ảnh thành mảng 64 64 3
test_image = np.expand_dims(test_image, axis = 0) #chuyển mảng 64 64 3 thành 1 64 64 3 trong đó 1 là số lượng ảnh đầu vào điều này rất quan trọng vì cnn muốn đầu vào là 1 batch
result = cnn.predict(test_image)
training_set.class_indices # dòng này chuyển đổi {'cat': 0, 'dog': 1}
if result[0][0] == 0:
  prediction = 'cat'
else:
  prediction = 'dog'


# In[45]:


print(prediction)


# In[ ]:




