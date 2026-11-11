# Vòng ba: nâng cấp đặc trưng HOG/HSV và MLP

Mốc đối chứng là mô hình thắng vòng hai: ảnh RGB 64×64 → HOG + histogram HSV
2268 chiều → chuẩn hóa theo thống kê train → MLP 256→64→32→6. Các tầng ẩn dùng
LayerNorm, ReLU và dropout 0,25. Loss CE, optimizer Adam LR 0,001. Accuracy test
trung bình đã đo là 77,32%, macro-F1 0,7741. Mọi mạng đều khởi tạo ngẫu nhiên.

## Các cấu hình được ấn định trước khi chạy

| Cấu hình | Cha | Thay đổi |
| --- | --- | --- |
| baseline | — | Chạy lại HOG + HSV → MLP thắng vòng hai |
| hog_only | baseline | Chỉ giữ HOG, bỏ toàn bộ HSV |
| hsv_only | baseline | Chỉ giữ HSV, bỏ toàn bộ HOG |
| highres | baseline | Tính HOG và HSV trên ảnh 128×128 |
| extra_lab | baseline | Ghép thêm histogram Lab theo lưới 1×1, 2×2, 4×4 |
| foreground_crop | baseline | Cắt bớt nền trắng bằng pipeline OpenCV cố định trước resize 64×64 |
| crop_highres | highres | Thêm cùng pipeline cắt nền trước resize 128×128 |
| gelu | baseline | Thay ReLU bằng GELU |
| light_dropout | baseline | Giảm dropout từ 0,25 xuống 0,10 |
| adamw_cosine | baseline | Dùng AdamW, weight decay 1e-4, cosine LR |

HOG-only và HSV-only giữ nguyên các tầng ẩn nhưng thay số đầu vào. Việc bỏ nhóm
đặc trưng cũng làm giảm số tham số, nên kết quả phản ánh cả biểu diễn và kích
thước lớp đầu. Độ phân giải cao tăng số chiều HOG; không phải so sánh cùng FLOPs.
Nhánh AdamW/cosine đổi một nhóm thiết lập, không tách riêng từng tác động.

## Biểu diễn đầu vào

HOG dùng block 16×16, stride 8×8, cell 8×8 và 9 hướng. Tại 64×64 có 1764 số;
tại 128×128 có 8100 số. Histogram màu có 8 bins/kênh trên 21 vùng của lưới
1×1, 2×2, 4×4, chia cho số pixel mỗi vùng. HSV có 504 số; thêm Lab có thêm 504 số.

| Biểu diễn | Số đầu vào MLP |
| --- | ---: |
| HOG + HSV ở 64×64 | 2268 |
| HOG-only | 1764 |
| HSV-only | 504 |
| HOG + HSV ở 128×128 | 8604 |
| HOG + HSV + Lab ở 64×64 | 2772 |

MLP baseline có 600294 tham số; HOG-only 471270; HSV-only 148710;
nhánh 128×128 có 2222310; nhánh thêm Lab có 729318. Các nhánh crop cùng độ phân
giải và các nhánh activation/dropout/optimizer giữ nguyên số tham số so với cha.

Mỗi descriptor được tính từ ảnh gốc. Đặc biệt, nhánh 128×128 không dùng ảnh đã
giảm xuống 64×64 rồi phóng lên. Ảnh được crop/resize trước, rồi lật ngang khi
augmentation; descriptor tính lại từ ảnh đã lật. Cùng hàm trích xuất được dùng
khi huấn luyện, validation, test và suy luận từ ảnh.

Mean/std mỗi đặc trưng chỉ được fit trên train chưa augmentation; std chặn dưới
ở 0,001. Hai buffer này lưu trong checkpoint và không cập nhật khi đánh giá.

## Pipeline cắt nền cố định

Thực hiện trên ảnh RGB gốc:

1. Tạo mask `gray < 235 OR saturation > 30`.
2. Morphological opening với kernel 3×3, sau đó closing với kernel 5×5.
3. Chọn connected component có diện tích lớn nhất, bỏ nền.
4. Nếu diện tích component ≥1% diện tích ảnh, lấy bounding box, thêm lề 10%
   kích thước cạnh lớn nhất và giới hạn trong ảnh. Pad trắng thành hình vuông.
5. Resize về độ phân giải descriptor. Nếu không có component đủ lớn, dùng ảnh gốc.

Ngưỡng được đặt trước, không fit từ validation/test hoặc nhãn. Cách này tận dụng
nền tương đối trắng của TrashNet, nhưng có thể bỏ sót vật thể sáng, trong suốt
hoặc phần tách rời. Không mặc định xem crop là cải tiến trước khi có kết quả.

## Quy tắc đánh giá và khả năng tái lập

- Manifest cố định: 1764 train, 380 validation, 377 test.
- Hai seed 42/43, mỗi seed một GPU T4; 600 epoch/cấu hình, batch 64, float32,
  deterministic, Xavier-normal; lật ngang p=0,5.
- Chọn checkpoint mỗi lần chạy theo validation macro-F1; accuracy phá hòa.
- Chọn cấu hình theo validation macro-F1 trung bình hai seed; accuracy phá hòa.
- Ghi `selection.json` trước khi nạp ảnh test; chỉ đánh giá test cho baseline
  và cấu hình thắng, không chọn seed bằng test.
- Tập test đã được xem trong các vòng trước. Đây là so sánh thăm dò, hai seed
  chưa đủ khẳng định ý nghĩa thống kê hoặc giới hạn của mô hình.

Thời gian trích xuất descriptor được ghi riêng; thời gian huấn luyện không bao
gồm bước này. Với cấu hình dùng lại descriptor đã cache, không xem thời gian cache
gần bằng 0 là chi phí trích xuất thực tế. Chuẩn hóa và descriptor là một phần của
pipeline suy luận, cần tính cả hai khi đo tốc độ ứng dụng.

## Chạy lại

```powershell
python -m wastelens.mlp_experiments train --suite round3 --seed 42 --epochs 600 --device cuda --output outputs/mlp-round3/seed-42
python -m wastelens.mlp_experiments train --suite round3 --seed 43 --epochs 600 --device cuda --output outputs/mlp-round3/seed-43
python -m wastelens.mlp_experiments summarize --suite round3 --device cuda --output outputs/mlp-round3
```

Đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi khởi động worker CUDA. Notebook
`mlp_ablation_round3_kaggle.ipynb` chứa snapshot mã có SHA-256, không phụ thuộc
vào một Git branch có thể bị thay đổi giữa quá trình chạy.

## Kết quả và xác minh

Đã hoàn tất 10 cấu hình ×2 seed ×600 epoch trên Kaggle T4 ×2, tổng 12000 epoch.
Job `job_c8cea751fa94447b833edc456844884e`, kernel
`luquang231/wastelens-mlp-round3-20261111`. Kết quả và biểu đồ trong
[báo cáo vòng ba](../reports/mlp-round3-20261111.md).

HOG + HSV + Lab được chọn bằng validation macro-F1 trung bình. Test accuracy
hai seed là 79,31% và 78,51%, trung bình 78,91%; macro-F1 trung bình 0,7774.
Baseline chạy lại giữ đúng kết quả vòng hai: accuracy 77,32%, macro-F1 0,7741.
Validation macro-F1 của nhánh thắng gần bằng HSV-only; không diễn giải thứ hạng
này thành một khác biệt có ý nghĩa thống kê.

Đã đối chiếu descriptor cũ với phiên bản nguồn vòng hai, kiểm tra tính đúng của
descriptor ảnh lật, kích thước đặc trưng, gradient hữu hạn, trường hợp crop ảnh
trắng/nhạt và pipeline suy luận từ ảnh gốc. Chạy thử 20 lượt ×2 epoch để kiểm tra
pipeline chọn mô hình, test và xuất báo cáo. Điểm của các lượt chạy thử không
được dùng để đánh giá chất lượng.

Kết quả đầy đủ được kiểm tra SHA-256 của snapshot và manifest, 20 checkpoint,
12000 dòng epoch trong lịch sử, đúng quy tắc chọn checkpoint và số tham số.
Checkpoint baseline của từng seed khớp từng tensor với mô hình HOG/HSV thắng
vòng hai. Khi nạp lại trên CPU, dự đoán và confusion matrix test của baseline
và mô hình thắng khớp Kaggle; hàm suy luận từ ảnh gốc khớp pipeline batch.
