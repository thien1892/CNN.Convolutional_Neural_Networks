# Tổng quan về CNN  
<img src ='https://i.imgur.com/jqceLIu.jpg'>  

# Cách khai báo thư viện, triển khai CNN:
**1. Khai báo thư viện:**  

```sh
from keras.layers import ZeroPadding2D, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten  
from keras.models import Model   
from keras.layers import Dense, Activation, Flatten
```

**2. Triển khai mô hình:**   
```sh
def model(input_shape):   
 X_input = Input(input_shape)   
 #zero padding   
 X = ZeroPadding2D((3,3))(X_input)  
 #Conv2D(filter, fxf, strides = (strides height, s width)):  
 X = Conv2D(32, (7,7), strides = (2,2))(X)
 X = Activation('relu')(X)
 #MaxPooling  
 X = MaxPooling2D((2,2))(X)
 X = Flatten()(X)
 X = Dense(1, activation= 'sigmoid')(X)

 model = Model(inputs = X_input, outputs = X, name = 'test')
 return model
```

**3. Bài tập thực hành:**  
**1. Thực hành với data mnist:**  
Ở phần [Thực hành NN với Data Mnist](https://github.com/thien1892/Thuc_hanh_voi_Neural_Network/blob/main/datamnist.ipynb) tôi đã sử dụng mạng Deep NN 3 lớp, khoan hãy nói tới độ chính xác cao hơn của mô hình CNN, ta hãy xem số tham số đào tạo của 2 mô hình:   
- Đối với Deep NN là 101.770 tham số, của CNN là 6.526 --> tầm 1/15 so với Deep NN!!!
- Mô hình tôi sử dụng CNN sẽ có dạng quen thộc với mô hình LENET (Trong thị giác máy tính, bạn sẽ sử dụng nhiều các mô hình học chuyển giao.)   
- Mô hình cụ thể của 2 phương pháp này như sau:  
-   ```sh  
    Loss = 0.04998834803700447
    Train Accuracy = 0.9859166741371155

    Loss = 0.09600713104009628
    Test Accuracy = 0.9733999967575073

    Model: "mnist"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 784)]             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    ```
-   ```sh
    Loss = 0.0431252084672451
    Train Accuracy = 0.9861000180244446

    Loss = 0.05485863611102104
    Test Accuracy = 0.9825000166893005  

    Model: "Mnist_CNN"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 6)         12        
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0         
    _________________________________________________________________
    batch_normalization (BatchNo (None, 14, 14, 6)         24        
    _________________________________________________________________
    activation (Activation)      (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 16)          0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 5, 5, 16)          64        
    _________________________________________________________________
    activation_1 (Activation)    (None, 5, 5, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 400)               0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                4010      
    =================================================================
    Total params: 6,526
    Trainable params: 6,482
    Non-trainable params: 44
    ```