# Minh chứng Lab03

| Yêu cầu | Thẻ | Minh chứng |
| --- | --- | --- |
| Dataset có ít nhất 3 lớp, mô tả và bản sao Drive công khai | WL-02, WL-12 | [Bảng phân bố 6 lớp](../README.md#trashnet-split), [manifest](../data/splits/trashnet.csv), liên kết bản sao Drive trong báo cáo |
| Nâng mô hình Lab02 từ nhị phân lên đa lớp | WL-03 | [Mô hình 6 logits](../src/wastelens/model.py), [đối chiếu kiến trúc](../README.md#lab02-to-lab03), cấu hình và checkpoint từ notebook |
| Đánh giá mô hình trên dataset | WL-04 | [Notebook huấn luyện và đánh giá](../notebooks/train_kaggle.ipynb) xuất accuracy, macro-F1, precision/recall theo lớp, confusion matrix và ảnh dự đoán sai |
| Giao diện và đầu vào ảnh, video | WL-05–WL-07 | Demo upload JPG, PNG và MP4 với giao diện thao tác được |
| Module 1 có điều khiển màu, hình học và làm mượt | WL-05 | Ảnh trước/sau cho từng nhóm phép xử lý và tham số điều khiển |
| Module 2 hiện Top-1, confidence, thời gian xử lý và xác suất mọi lớp | WL-06, WL-07 | Ảnh chụp kết quả và biểu đồ 6 lớp, clip xử lý MP4 |
| Snapshot và export | WL-08 | Ảnh và video xuất ra được mở lại trong trình xem độc lập |
| Phân tích kết hợp filter OpenCV với mô hình | WL-09 | Bảng có đối chứng và nhận xét ảnh hưởng của filter lên chỉ số |
| Phân tích binary và multi-class | WL-10 | Bảng so sánh loss, gradient, cập nhật trọng số và độ phức tạp |
| Sơ đồ kiến trúc, ảnh giao diện và video demo | WL-11, WL-12 | Sơ đồ từ đầu vào đến UI, ảnh từng chức năng và link demo mở được khi không đăng nhập |
| Báo cáo, bảng tự chấm và gói mã nguồn | WL-11, WL-13 | Bảng tự chấm 0–100% theo yêu cầu, pseudo-code, báo cáo nguồn và PDF trong `doc/`, mã có chú thích trong `source/` |
| Webcam real-time (bonus) | WL-14 | Demo nhận diện webcam bằng ROI cố định nếu thực hiện |

Khi viết báo cáo, điền mức tự chấm theo chức năng đã chạy thử và dẫn tới minh chứng tương ứng. Kiểm tra hai liên kết công khai trước khi đóng gói và giữ nguyên sau hạn nộp chính thức.
