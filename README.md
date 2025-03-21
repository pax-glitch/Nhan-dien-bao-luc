Hệ Thống Phát Hiện Bạo Lực Từ Video Streaming

Giới Thiệu

Hệ thống này sử dụng camera để phát hiện hành vi bạo lực trong thời gian thực bằng cách kết hợp các kỹ thuật nhận diện đối tượng, phân tích chuyển động và mô hình học sâu. Nếu phát hiện bạo lực, hệ thống sẽ kích hoạt cảnh báo và ghi lại sự kiện vào cơ sở dữ liệu MongoDB, đồng thời lưu video để xem lại sau này.

Các Công Nghệ Sử Dụng

OpenCV: Xử lý hình ảnh và video. TensorFlow/Keras: Sử dụng mô hình học sâu để phân tích hành vi. YOLOv8: Nhận diện đối tượng (người) trong video. MoveNet (TFLite): Nhận diện tư thế cơ thể. MongoDB: Lưu trữ dữ liệu các sự cố. Pygame: Phát âm thanh cảnh báo.

Nguyên Lý Hoạt Động

Lấy dữ liệu từ camera: Camera được kết nối qua IP từ điện thoại Android. Phát hiện người bằng YOLO: Hệ thống xác định vị trí con người trong khung hình. Nhận diện tư thế bằng MoveNet: Dự đoán vị trí các bộ phận cơ thể. Phân tích chuyển động: Đánh giá tốc độ, độ giật và tư thế bất thường. Phát hiện bạo lực: Sử dụng mô hình học sâu để xác định xem có hành vi bạo lực không. Cảnh báo & ghi hình: Nếu phát hiện bạo lực, hệ thống sẽ: Phát cảnh báo âm thanh. Ghi lại video sự cố trước và sau khi xảy ra. Lưu thông tin sự cố vào MongoDB. Người dùng có thể xem lại video đã ghi.

Hướng Dẫn Cài Đặt Cài đặt các thư viện cần thiết: pip install opencv-python numpy torch tensorflow ultralytics pymongo pygame

Chạy chương trình: python phat_hien_bao_luc.py

Cấu Hình MongoDB Bạn cần thay đổi MONGODB_CONNECTION_STRING trong code để phù hợp với hệ thống của bạn.

Lưu Ý Điện thoại và máy tính phải cùng mạng WiFi. Đảm bảo URL camera IP đúng (ví dụ: http://192.168.x.x:4747/video). Nếu phát hiện sai, có thể điều chỉnh ngưỡng violence_threshold trong cấu hình.
