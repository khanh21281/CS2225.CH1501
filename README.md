# CS2225.CH1501 - Final Project
# Xác định một người có mang khẩu trang hay không, đồng thời cho biết họ tên người đó từ camera quan sát. 

## **Thành viên nhóm**
<img src="https://i.ibb.co/wYNJBF4/Microsoft-Teams-image.png)">

## **Giới thiệu về đồ án**
- Trong đại dịch Covid-19, mọi người được khuyến cáo phải mang khẩu trang nhằm tự bảo vệ mình và ngăn chặn virus lây lan. Tuy nhiên vẫn còn một số người bỏ qua khuyến cáo này.
- Nhóm đã xây dựng một ứng dụng machine learning nhằm phát hiện những người không mang khẩu trang, lưu lại hình ảnh kèm theo họ tên của họ vào máy tính theo thời gian thực. 
- Ứng dụng được viết bằng python và một số thư viện như OpenCV, Keras..
## **Hình minh họa**

## **Mô tả bài toán**
- Task: 
  - Dự đoán việc mang khẩu trang.
  - Dự đoán họ tên.
  - Lưu lại hình ảnh.
- Input: 
  - Hình ảnh từ camera quan sát.
- Output:
  - Vị trí các khuôn mặt và họ tên kèm theo.
  - Bounding box màu đỏ nếu không mang khẩu trang, nếu có mang thì bounding box màu xanh.
  - Hình ảnh được lưu lại trên máy tính.

## **Kiến thức nền tảng**
### 1. Convolutional Neural Networks (CNN)
- Convolutional Neural Networks (mạng noron tích chập) chỉ ra rằng mạng sử dụng một phép toán được gọi là tích chập. Convolutional networks là một loại mạng nơron chuyên biệt sử dụng tích chập thay cho phép nhân ma trận chung trong ít nhất một trong các lớp của chúng.
- Một mạng CNN bao gồm một lớp đầu vào và một lớp đầu ra, ở giữa gồm nhiều lớp ẩn. Các lớp ẩn của CNN thường bao gồm một loạt các lớp phức hợp có thể thay đổi theo phép nhân hoặc tích số dot khác. Activation Function (hàm kích hoạt) thường là một lớp ReLU, và sau đó được theo sau bởi các phần chập bổ sung như các lớp pooling, fully connected và normalization, được gọi là các lớp ẩn vì các đầu vào và đầu ra của chúng bị che bởi hàm kích hoạt và tích chập cuối cùng. CNN chủ yếu được sử dụng để nhận dạng hình ảnh vì phương pháp này có nhiều lợi thế hơn so với các phương pháp khác.

### 2. Feature extraction
- CNNs lấy một hình ảnh đầu vào và chuyển đổi nó thành các feature vectors (vectơ đặc trưng). Các feature vectors (còn được gọi là embeddings hay bottleneck features) về cơ bản là một tập hợp của một vài nghìn giá trị floating-point. Đi qua các lớp convolution và pooling trong CNN về cơ bản là một hành động giảm bớt, để lọc thông tin có trong hình ảnh thành các thành phần nổi bật và quan trọng nhất của nó, từ đó hình thành các bottleneck features.

### 3. Image processing
- Xử lý ảnh là một trong những phần chính của cấu trúc xử lý tín hiệu, trong đó đầu vào ở dạng hình ảnh; ví dụ, một bức ảnh hoặc video clip. Hiệu suất xử lý hình ảnh cũng có thể là một hình ảnh hoặc nhiều đặc điểm hoặc thông số liên quan đến hình ảnh. Mục tiêu chính của tiền xử lý hình ảnh là nâng cấp thông tin hình ảnh nơi các hình ảnh thừa được lấy ra và các hình ảnh quan trọng được thêm vào để xử lý bổ sung.
- Xử lý hình ảnh, hoặc là một cải tiến cho người xem hoặc phần mềm phân tích tự động mang lại lợi thế về tính linh hoạt, tốc độ và chi phí.

### 4. Face detection
