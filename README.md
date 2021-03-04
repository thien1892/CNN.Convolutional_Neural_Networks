# CNN_Convolutional_Neural_Networks
## Tổng quan về CNN
1. [Tổng quan về CNN](https://github.com/thien1892/CNN_Convolutional_Neural_Networks/blob/main/Tong_quan_CNN.md)  
2. [Bài tập thực hành nhận diện chữ số viết tay mnist_cnn](https://github.com/thien1892/CNN_Convolutional_Neural_Networks/blob/main/mnist_cnn.ipynb)   
3. [Bài tập thực hành nhận diện ký hiệu số từ bàn tay](https://github.com/thien1892/CNN_Convolutional_Neural_Networks/blob/main/signs_cnn.ipynb)  

## Các mô hình CNN:
### 1. Mô hình cơ bản:  
#### 1.1. LeNET-5   
- Nội dung: [lecun et al. 1998. gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)  
- Mô hình cụ thể:  
<img src = 'https://i.imgur.com/qG3vgSo.jpg'>      

- Nhận xét:  
    - Mô hình có khoảng 60.000 tham số, với mô hình CNN hiện nay thì đây là 1 mô hình nhỏ.  
    - Đi từ phải qua trái ta thấy CONV -> POOL -> CONV -> POOL -> FC -> FC -> y_hat; ra đời từ 1998,và cách sau mỗi CONV có POOL vẫn còn áp dụng ở các mô hình sau này.  
    - Một số thay đổi hiện nay: thời đó người ta thường dùng các hàm phi tuyến: sigmoid / tanh thay vì relu, trong bài báo sau mỗi CONV là 1 hàm phi tuyến rồi mới đến POOL; để tiết kiệm tính toán, người ta cũng không dùng số lớp fxf bằng n'c mà sử dụng một cách phức tạp hơn; hiện nay chúng ta hay dùng MaxPooling hơn AveragePooling.

#### 1.2. AlexNet   
- Nội dung: [krizhevsky et al. 2012. imagenet classification with deep convolutional neural networks](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf)  
- Mô hình cụ thể: 
<img src = 'https://i.imgur.com/N43XbGU.jpg'>  

- Nhận xét:
    - AlexNet tương tự với LeNET-5 nhưng lớn hơn, tầm 60.000.000 tham số.
    - Nó sử dụng RELU; các layer nh, nw không thay đổi sử dụng padding same
#### 1.3. VGG (VGG-16)
- Nội dung: [Karen Simonyan, Andrew Zisserman 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) 
- Mô hình cụ thể: 
<img src = 'https://i.imgur.com/dxMMViz.jpg'>

- Nhận xét:
    - Đây là mô hình lớn với khoảng 138.000.000 tham số, mô hình có tính hấp dẫn với mỗi CONV (3x3, s =1, padding same), nh,nw sau một số lần CONV giảm xuống, nc tăng từ 64, 128, 256, 512.
    - Tai sao VGG-16? vì có 16 layer chứa trọng số, một mô hình khác tương tự là VGG-19. Trên thực tế VGG-16 và VGG-19 cho kết quả gần như nhau.
### 2. Resnet   
- Nội dung: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- Resnet được xây dựng dựa trên các khối dư (residual block), cụ thể như sau: 
    <img src = 'https://i.imgur.com/Nzzxaps.jpg'> 
- Resnet network sẽ có hình dạng sau: 
    <img src = 'https://i.imgur.com/LHTqDAy.jpg'>
    - Các mô hình không có residual block, khi đào tạo các lớp sâu hơn đến một lúc nào đó trên thực tế sẽ làm gia tăng lỗi! Lý do là các mô hình sâu hơn, đào tạo khó khăn hơn --> lỗi sinh ra do huấn luyện làm cho thực tế mô hình càng tệ hơn.
    - Với mô hình resnet các lớp sâu hơn sẽ đảm bảo làm giảm lỗi huấn luyện.
- Tại sao Resnet hoạt động tốt?
    <img src = 'https://i.imgur.com/NEzuTKL.jpg'>
    - Ví dụ ta thêm 1 khối residual block: 2 layer vào một mô hình đủ lớn, khi đó nếu chúng ta sử dụng các phương pháp để đào tạo tham số nếu w[l+1], b[l+1] biến mất thì a[l+2] = g(a[l]) = a[l] --> Do đó việc thêm 1 khối residual block không ảnh hưởng đến mô hình. Và nếu các lớp a[l+1], a[l+2] học hỏi được điều gì hữu ích thì hiệu suất mô hình sẽ tăng lên!!!
    - Khi a[l+2] và a[l] có kích thước khác nhau, ta thêm Ws có kích thước phù hợp --> a[l+2] = g(z[l+2] + Ws x a[l]).
- Hình ảnh mô hình Resnet-34 (đọc trong bài báo thì mô hình này phát triển từ mô hình VGG-19):
    <img src = 'https://i.imgur.com/zcoV5kn.jpg'>

### 3. Inception
- [Network in network (Conv 1x 1)](https://arxiv.org/pdf/1312.4400v3.pdf)
    <img src = 'https://i.imgur.com/kLk5wT0.jpg'> 
    - Conv 1x 1 có ý nghĩa không?, xem hình ảnh trên, khác với phép nhân 2 chiều đơn giản, conv 1x1 là phép nhân từng lát cắt nhỏ, kết hợp với 1 hàm phi tuyến (ví dụ relu)--> giảm sự phức tạp của layer đang xem xét.
    - Conv 1x1 làm tăng giảm nc, bảo toàn nh, nw
    - Conv áp dụng trong nhiều mô hình CNN, ví dụ: Inception.
    - Conv1x1 giống như 1 đèn pin thu nhỏ tăng giảm kênh --> trong quá trình tăng giảm giúp tối ưu hóa việc tính toán.
- Động lực của mô hình Inception:
    - Ta mong muốn áp dụng cùng lúc nhiều bộ lọc(1x1, 3x3, 5x5) cũng như Pooling, dù cho việc này sẽ làm mô hình trở nên phức tạp, nhưng đổi lại sẽ tăng khả năng hiệu suất của mô hình. Inception là mô hình như vậy.
    <img src ='https://i.imgur.com/QmjVevi.jpg'> 
    - Áp dụng CONV 1x1 một cách hợp lý giúp tối ưu hóa chi phí tính toán và không làm giảm hiệu năng mô hình, khi không có CONV 1x1, chi phí tính toán của bộ lọc 5x5 trong Inception là 120.000.000 phép nhân; còn khi áp dụng Conv1x1, chi phí tính toán chỉ còn 12.400.000 phép tính!!!
    <img src = 'https://i.imgur.com/zAUKJ9I.jpg'> 
    <img src = 'https://i.imgur.com/TL4dLSO.jpg'>
- Module Inception:
    <img src = 'https://i.imgur.com/tR9eews.jpg'>
- [Inception Network](https://arxiv.org/pdf/1409.4842.pdf):
    <img src ='https://i.imgur.com/ehN6NVr.jpg'>
    - Mạng Inception ở các layer trung gian cũng giúp chúng ta kích hoạt output, việc thử các kích hoạt này đảm bảo mô hình overfitting.
    - "We need to go deeper!", inception lấy cảm hứng từ câu thoại trong phim inception!!! Chúng ta nên đi sâu hơn!!! (miễn là đừng để quá khớp)

### 4. Thực hành:  
- Thực hành Data Sign với mô hình Resnet50:
    - Dowload Model [ResNet50.h5](https://drive.google.com/file/d/189RVfM-u_pV7nOwQ1wEKtuOT8oz2A5uM/view?usp=sharing)




