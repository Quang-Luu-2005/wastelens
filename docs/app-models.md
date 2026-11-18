# Ba mô hình cho application

Application sử dụng ba checkpoint đã huấn luyện. Người dùng chọn rõ mô hình để
dự đoán; chế độ so sánh chạy độc lập cả ba trên cùng ảnh sau Module 1. Không trộn
MobileNet vào ensemble MLP và không chọn kết quả theo confidence cao nhất.

## Kiến trúc và vai trò

| | Baseline kế thừa Lab 2 | MLP cải tiến | MobileNetV3 Large |
| --- | --- | --- | --- |
| Vai trò | Mốc so sánh thay đổi từ phân lớp nhị phân sang 6 lớp | Phương án cải tiến từ đầu, mặc định trong app | Đối chiếu với transfer learning |
| Đầu vào | RGB 64×64, flatten 12.288 giá trị | HOG + HSV + Lab (2.772 giá trị) hoặc HSV (504 giá trị) | RGB 224×224, chuẩn hóa ImageNet trong model |
| Kiến trúc | 12.288 → 128 → 32 → 6 | Mỗi MLP: đầu vào → 256 → 64 → 32 → 6, LayerNorm, ReLU, dropout 0,25 | MobileNetV3 Large với đầu ra thay bằng 6 lớp |
| Trọng số ban đầu | Ngẫu nhiên | Ngẫu nhiên; đặc trưng OpenCV không có trọng số pretrained | ImageNet pretrained, fine-tuning toàn mạng |
| Số tham số | 1.577.318 | 4.390.140 cho tổng cộng 10 MLP | 4.209.718 |
| Suy luận | Softmax logits | Trung bình xác suất 5 MLP HOG/HSV/Lab và 5 MLP HSV, rồi temperature scaling T=2 | Softmax logits |
| Confidence | Chưa hiệu chỉnh | Đã hiệu chỉnh; dưới 0,5258051753 hiển thị “Không chắc” | Chưa hiệu chỉnh |

Baseline giữ **hai tầng ẩn 128 và 32** của Lab 2. “6 lớp” là sáu lớp phân loại
ở đầu ra, không phải thêm sáu tầng ẩn hoặc đổi đầu vào thành 6. Đầu ra cũ
32→1 với sigmoid/BCE được thay bằng 32→6 với softmax/cross-entropy. Lab 3 dùng
autograd và Adam để cập nhật trọng số; không tái sử dụng trọng số Horse or Human.

MLP cải tiến tăng số tầng ẩn lên ba, thay biểu diễn đầu vào, thêm chuẩn hóa và
regularization. Bản triển khai là **ensemble 10 mạng**, không phải một mạng duy
nhất. Việc kết hợp dự đoán và hiệu chỉnh xác suất là bước suy luận sau huấn luyện.
MobileNet là pretrained rồi fine-tuning; hai phương án MLP là build from scratch.

## Checkpoint và lựa chọn huấn luyện

| ID | File nguồn trong `outputs/` | File dùng trong `models/` | Lựa chọn checkpoint |
| --- | --- | --- | --- |
| `baseline` | `kaggle-t4-600ep-20261110/seed-43/best.pt` | `wastelens-baseline.pt` | Seed 43 chọn theo validation macro-F1; epoch 342 của 600, val macro-F1 0,580082 |
| `improved` | `kaggle-ensemble-600-20261113/ensemble-300.pt` | `wastelens-ensemble.pt` | 5 seed 42–46 cho mỗi biểu diễn, mỗi mạng 300 epoch; cấu hình dựa trên các thử nghiệm CV trước đó |
| `mobilenet` | `kaggle-mobilenetv3-100ep-20261110/seed-43/best.pt` | `wastelens-mobilenetv3.pt` | Seed 43 chọn theo validation macro-F1; epoch 62 của 100, val macro-F1 0,958346 |

SHA-256 của ba file theo thứ tự trên:

```text
baseline  4bcfae31391cdf900454785e10cd6345ea87716f22e1c0c82e7fa6add70aa8f1
improved  061513298ee3991b4c3de62f5ce7d2ff20c1907b8ce9e2803d0418ffe6880d45
mobilenet dd62a1e6500111cc02547bac4a1c96f386b9514fdb6f1b82f031ef7706203c36
```

Sau khi tải các output về đúng đường dẫn trong repo, chạy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_app_models.py
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Script kiểm tra hash cả ba nguồn trước khi sao chép và từ chối ghi đè một file
đích khác hash. Có thể chép trực tiếp các file đã xác minh vào `models/` nếu
không giữ thư mục output gốc. Weights không đưa vào Git. Nạp checkpoint MobileNet
đã lưu không tải thêm trọng số ImageNet hoặc cần kết nối mạng.

## Kết quả kiểm tra lại qua app

Ngày 18/11/2026, chạy `scripts/evaluate_app_models.py` trên CPU với đúng
377 ảnh test từ `data/splits/trashnet.csv`, không augmentation. Đường suy luận
nhận ảnh OpenCV BGR được đối chiếu với PIL và API `predict_probabilities()`.

| Model | Đúng/tổng | Accuracy toàn test | Macro-F1 |
| --- | ---: | ---: | ---: |
| Baseline | 221/377 | 58,62% | 0,572651 |
| MLP cải tiến | 323/377 | 85,68% | 0,860745 |
| MobileNetV3 Large | 357/377 | 94,69% | 0,946537 |

Chỉ số trên tính Top-1 trên **toàn bộ test**, kể cả ảnh bị ensemble đánh dấu
“Không chắc”. Riêng ensemble chấp nhận 300/377 ảnh (coverage 79,58%), đúng
278/300 ảnh (92,67%). Không dùng 92,67% để thay accuracy toàn test trong bảng.

Baseline và MobileNet dùng 1.764 ảnh train, 380 ảnh validation. Ensemble được
huấn luyện lại trên 2.144 ảnh train+validation sau CV. Vì vậy bảng phản ánh các
checkpoint hiện có, không phải ablation chỉ thay kiến trúc với mọi điều kiện
huấn luyện giữ nguyên. Test đã được xem trong nhiều vòng thử nghiệm; kết quả
mang tính thăm dò. Chạy so sánh trong app trên một ảnh cũng không đo accuracy.

JSON xuất từ app ghi rõ model ID, tên checkpoint, cách huấn luyện, loại
confidence và ngưỡng. Khi bật so sánh, JSON chứa đủ xác suất sáu lớp và thời gian
của từng mô hình. Snapshot luôn là kết quả của mô hình đang chọn.
