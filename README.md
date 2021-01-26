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
