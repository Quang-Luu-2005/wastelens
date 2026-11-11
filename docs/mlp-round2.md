# Vòng hai: cải tiến biểu diễn đầu vào và kiến trúc MLP

Mốc đối chứng là MLP thắng vòng một: các tầng ẩn 256→64→32, LayerNorm trước
ReLU, dropout 0,25, CE và Adam LR 0,001. Mốc test trung bình đã đo là 62,33%.
Vòng này tiếp tục dùng mạng tự khởi tạo, không sử dụng MobileNet, CNN pretrained
hay trọng số học từ dataset khác.

| Cấu hình | Cấu hình cha | Thay đổi |
| --- | --- | --- |
| baseline | — | Chạy lại mô hình thắng vòng một |
| no_dropout | baseline | Bỏ dropout, giữ LayerNorm |
| channel_std | baseline | Chuẩn hóa từng kênh RGB bằng mean/std của train |
| pixel_std | baseline | Chuẩn hóa từng tọa độ pixel bằng mean/std của train |
| gelu | channel_std | Thay ReLU bằng GELU |
| light_dropout | gelu | Giảm dropout từ 0,25 xuống 0,10 |
| plain_blocks | light_dropout | Stem 256 neuron và hai block dense, mỗi block hai Linear 256→256, không skip |
| residual_blocks | plain_blocks | Thêm phép cộng đầu vào với đầu ra của mỗi block |
| hog_hsv | baseline | Thay pixel bằng vector HOG + HSV chuẩn hóa theo train |
| hybrid | baseline | Ghép pixel và HOG + HSV, chuẩn hóa theo train |

Hai mạng plain/residual có cùng tham số, cùng khởi tạo và cùng phép biến đổi
trong nhánh chính. Phép cộng skip là khác biệt duy nhất giữa hai cấu hình này.
So sánh `light_dropout` với `plain_blocks` là thay đổi cả bố trí và độ sâu của mạng,
không được quy toàn bộ chênh lệch cho số tầng.

## Chuẩn hóa và đặc trưng cố định

Thống kê chỉ được fit trên ảnh train chưa augmentation. Mean và std được lưu
trong checkpoint dưới dạng buffer, không cập nhật khi validation, test hay suy
luận. Std của pixel/kênh được chặn dưới ở 0,05 để tránh khuếch đại vùng nền gần
như hằng số. Std của descriptor được chặn dưới ở 0,001.

Descriptor dùng OpenCV trên RGB 64×64:

- HOG tính từ ảnh xám với ô 8×8, block 16×16, stride 8×8, 9 hướng, tạo 1764 số.
- HSV có histogram 8 bins cho mỗi kênh, trên lưới 1×1, 2×2 và 4×4. Mỗi histogram
  chia cho số pixel của vùng, tổng cộng 504 số.
- HOG + HSV có 2268 đầu vào. Hybrid ghép thêm 12288 pixel RGB, thành 14556 đầu vào.

Đây là các phép trích xuất đặc trưng cố định, không có trọng số pretrained. MLP
học ánh xạ từ descriptor sang 6 lớp. Gradient cập nhật các trọng số MLP; không
backprop qua bộ descriptor OpenCV. Đây là một nhánh thay đổi biểu diễn đầu vào,
cần trình bày riêng với nhánh MLP trực tiếp trên pixel.

Với augmentation lật ngang, descriptor được tính lại từ ảnh đã lật. Đảo ngược
vector descriptor không tương đương với lật ảnh. Pipeline suy luận dùng cùng
hàm trích xuất, cùng thứ tự đặc trưng và cùng buffer chuẩn hóa như huấn luyện.

## Quy tắc thí nghiệm

Giữ manifest cũ: 1764 train, 380 validation, 377 test. Chạy seed 42 và 43, 600
epoch mỗi cấu hình, batch 64, float32 deterministic, Xavier-normal, CE, Adam
LR 0,001 cố định. Chỉ dùng lật ngang p=0,5. Không đổi loss, scheduler hoặc cách
chia dữ liệu trong vòng này.

Chọn epoch theo validation macro-F1 rồi accuracy nếu hòa. Chọn cấu hình bằng
trung bình validation macro-F1 của hai seed. Ghi quyết định trước khi nạp ảnh
test, sau đó đánh giá test cho baseline và cấu hình thắng. Tập test này đã được
xem ở những vòng trước, nên kết quả vẫn mang tính thăm dò. Hai seed chưa đủ để
khẳng định ý nghĩa thống kê hoặc tìm được giới hạn tối ưu.

Thời gian `training_seconds` không bao gồm việc trích xuất descriptor đã cache.
Số tham số và thời gian trích xuất đặc trưng cần được cân nhắc khi so sánh chi phí.

| Nhóm mạng | Số đầu vào MLP | Số tham số học |
| --- | ---: | ---: |
| MLP pixel thông thường | 12288 | 3.165.414 |
| Plain/residual blocks | 12288 | 3.412.742 |
| HOG + HSV → MLP | 2268 | 600.294 |
| Pixel + HOG + HSV → MLP | 14556 | 3.746.022 |

## Chạy lại

```powershell
python -m wastelens.mlp_experiments train --suite round2 --seed 42 --epochs 600 --device cuda --output outputs/mlp-round2/seed-42
python -m wastelens.mlp_experiments train --suite round2 --seed 43 --epochs 600 --device cuda --output outputs/mlp-round2/seed-43
python -m wastelens.mlp_experiments summarize --suite round2 --device cuda --output outputs/mlp-round2
```

Đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi khởi động worker CUDA. Tạo notebook
tự chứa snapshot mã bằng `scripts/build_mlp_kaggle.py --suite round2 --staging PATH`.
Notebook chạy hai seed trên hai GPU T4 và lưu ZIP nguồn kèm SHA-256 vào output.

## Kết quả và xác minh

Đã hoàn tất trên Kaggle T4 ×2, job `job_7ba5fa2818404786a2d30a6327fe4628`,
kernel `luquang231/wastelens-mlp-round2-20261111`. Mỗi trong 20 lần chạy có đủ
600 epoch, tổng 12000 epoch. Kết quả và biểu đồ được lưu trong
[báo cáo vòng hai](../reports/mlp-round2-20261111.md).

Đã xác minh SHA-256 của snapshot nguồn và manifest, mọi checkpoint được chọn
đúng theo lịch sử validation, cùng số tham số được báo cáo. Checkpoint đối
chứng của từng seed khớp từng tensor với mô hình thắng vòng trước. Khi nạp lại
trên CPU, toàn bộ dự đoán và confusion matrix test của baseline và HOG/HSV
khớp kết quả Kaggle; hàm suy luận từ ảnh gốc cũng đã đối chiếu với pipeline batch.

Kết quả test trung bình của cấu hình chọn bằng validation là 77,32%, macro-F1
0,7741. Hai seed lần lượt đạt 77,45% và 77,19% accuracy. Không đánh giá test
cho các ứng viên còn lại trong vòng này.
