# Thí nghiệm nâng cấp MLP Lab02 cho TrashNet

Mục tiêu là kiểm tra các thay đổi cụ thể đối với mạng fully connected của Lab02,
khởi tạo ngẫu nhiên và huấn luyện trên 6 lớp rác. Các thí nghiệm trong tài liệu
này không sử dụng backbone hay trọng số pretrained.

## Thiết kế đối chứng

| Tên | Các tầng ẩn | Thay đổi so với cấu hình cha |
| --- | --- | --- |
| baseline | 128 → 32 | MLP Lab02 với đầu ra 6 logits, CE, Adam |
| deeper | 128 → 64 → 32 | Thêm một tầng ẩn so với baseline |
| wider | 256 → 64 → 32 | Tăng tầng ẩn đầu từ 128 lên 256 so với deeper |
| dropout | 256 → 64 → 32 | Thêm dropout 0,25 sau mỗi ReLU so với wider |
| layernorm | 256 → 64 → 32 | Thêm LayerNorm trước mỗi ReLU so với dropout |
| adamw_cosine | 256 → 64 → 32 | Từ layernorm, dùng AdamW, weight decay 1e-4 và cosine LR |
| label_smoothing | 256 → 64 → 32 | Từ adamw_cosine, đổi CE sang label smoothing 0,05 |
| weighted_ce | 256 → 64 → 32 | Từ adamw_cosine, dùng CE có trọng số lớp |
| focal | 256 → 64 → 32 | Từ adamw_cosine, dùng focal loss gamma=2 |

Nhánh `adamw_cosine` thay đổi một nhóm thiết lập tối ưu hóa cùng lúc, nên kết quả
của nó không tách riêng được tác động của AdamW, weight decay và cosine decay.
Ba nhánh loss đều so sánh với cùng cấu hình cha `adamw_cosine`.

Các cấu hình sử dụng cùng RGB 64×64, pixel [0,1], ReLU, Xavier-normal, batch 64,
learning rate khởi đầu 0,001 và tối đa **600 epoch**. Mỗi cấu hình chạy đủ 600
epoch, không early stopping. Augmentation chỉ là lật ngang với xác suất 0,5.
Giữ nguyên manifest train/validation/test đã dùng trước đây, lần lượt 1764/380/377
ảnh. Trọng số lớp chỉ tính từ nhãn train.

Hai seed 42 và 43 chạy trên hai GPU T4. Mỗi seed dùng bộ sinh số riêng cho thứ tự
batch và phép lật, để cùng seed có cùng thứ tự dữ liệu và augmentation giữa các
cấu hình. Dropout có nguồn ngẫu nhiên của mô hình riêng. Dùng float32, không AMP,
không TF32 và bật thuật toán deterministic. Baseline cũ có thể dùng AMP và một
thứ tự augmentation khác, vì vậy đối chứng chính là baseline chạy lại trong
chính thí nghiệm này, không lấy chênh lệch với một lần chạy lịch sử làm kết luận.

## Loss, gradient và cập nhật trọng số

Với C=6, logits z, p=softmax(z), nhãn y và batch N:

- CE thường có gradient theo logits là `(p - one_hot(y)) / N`.
- Label smoothing thay nhãn đích bằng `q = 0,95 * one_hot(y) + 0,05 / 6`,
  gradient thành `(p - q) / N`. Mục đích là giảm mức tự tin quá cao của mô hình;
  hiệu quả thực tế phải dựa trên kết quả đối chứng.
- Weighted CE dùng `w_c ∝ 1 / sqrt(n_c)`, trong đó `n_c` là số ảnh train của lớp c.
  Với reduction mean của PyTorch, gradient mẫu i được nhân
  `w_yi / sum_j(w_yj)` thay cho `1/N`. Mục đích là tăng đóng góp của lớp ít mẫu,
  nhưng macro-F1 tốt hơn không đảm bảo accuracy tổng cao hơn.
- Focal loss dùng `-(1-p_y)^2 * log(p_y)`, lấy trung bình trên batch. Cả hệ số
  điều biến và log-probability đều tham gia autograd; không detach hệ số. Loss
  này giảm đóng góp của mẫu đã được phân loại dễ, nhưng có thể tập trung vào
  mẫu nhiễu nên phải đánh giá thực nghiệm.

Autograd truyền gradient qua các tầng mới và cũ. Adam được giữ cho các nhánh
đầu. Nhánh AdamW dùng weight decay tách khỏi gradient của loss, với learning
rate giảm dần theo cosine. Sự khác nhau giữa tự viết gradient ở Lab02 và dùng
autograd ở Lab03 cần được mô tả trong báo cáo.

## Độ phức tạp

| Kiến trúc | Số tham số |
| --- | ---: |
| baseline | 1.577.318 |
| deeper | 1.583.526 |
| wider, dropout | 3.164.710 |
| layernorm và các nhánh sau | 3.165.414 |

Một tầng dense có kích thước d_in × d_out cần khoảng d_in*d_out phép nhân-cộng
cho mỗi mẫu ở forward. Lớp đầu chiếm phần lớn chi phí của MLP ảnh phẳng. Thêm
tầng 128→64→32 chỉ tăng nhẹ số tham số, trong khi tăng tầng đầu lên 256 gần như
nhân đôi tổng tham số. Backward có cùng bậc độ phức tạp với forward; optimizer
Adam/AdamW còn lưu hai trạng thái moment cho mỗi tham số. Dropout không thêm
tham số; LayerNorm thêm scale và bias cho từng neuron ở mỗi tầng ẩn.

`training_seconds` đo vòng huấn luyện, đánh giá validation mỗi epoch và sao chép
checkpoint tốt nhất, không bao gồm tải dataset và đánh giá test. Các ứng viên
có cùng ngân sách epoch, không có cùng ngân sách FLOPs. Bộ nhớ GPU đỉnh còn chứa
dataset đã cache trên GPU nên không được diễn giải thành kích thước mô hình.

## Quy tắc chọn và đánh giá

1. Trong mỗi lần chạy, chọn epoch theo validation macro-F1, accuracy phá hòa.
2. Chọn cấu hình theo trung bình validation macro-F1 trên hai seed, accuracy
   trung bình phá hòa. Báo cáo cả accuracy và macro-F1 cùng độ lệch chuẩn mẫu.
3. Ghi `selection.json` trước khi nạp ảnh test.
4. Chỉ đánh giá test cho baseline và cấu hình thắng, với cả hai seed. Báo cáo
   trung bình và từng seed, không chọn seed bằng điểm test.

Hai seed chưa đủ để khẳng định ý nghĩa thống kê hoặc giới hạn tối ưu của MLP.
Tập test này đã được xem trong các thí nghiệm trước của dự án, nên đây là so sánh
thăm dò, không phải một tập holdout mới chưa từng sử dụng. Kết quả thấp hơn cũng
là kết quả cần giữ lại, không loại cấu hình khỏi bảng vì không cải thiện.

## Chạy lại

```powershell
python -m wastelens.mlp_experiments train --output outputs/mlp-run/seed-42 --seed 42 --epochs 600 --device cuda
python -m wastelens.mlp_experiments train --output outputs/mlp-run/seed-43 --seed 43 --epochs 600 --device cuda
python -m wastelens.mlp_experiments summarize --output outputs/mlp-run --seeds 42 43 --device cuda
```

Khi bật deterministic trên CUDA, đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi
khởi động Python. Notebook `notebooks/mlp_ablation_kaggle.ipynb` đã thiết lập biến
này cho worker, chạy mỗi seed trên một GPU và chứa snapshot nguồn có SHA-256.

Tạo notebook từ mã hiện tại bằng:

```powershell
python scripts/build_mlp_kaggle.py --staging outputs/kaggle-mlp-stage-new
```

Control Plane yêu cầu thư mục nộp nằm trong thư mục `experiments/` được cho phép.
Sao chép notebook và `kernel-metadata.json` vào đó rồi nộp bằng tài khoản đã chọn.
Mỗi notebook lưu snapshot nguồn vào kết quả để kiểm tra được chính xác mã đã chạy.

## Kiểm tra triển khai

Đã kiểm tra baseline có cùng khởi tạo và đầu ra với `Lab02MLP` ở cùng seed;
cả chín nhánh có gradient hữu hạn, lưu/nạp checkpoint giữ nguyên đầu ra và hàm
suy luận trả sáu xác suất có tổng bằng một. Gradient focal loss được đối chiếu
sai phân hữu hạn bằng `torch.autograd.gradcheck`. Đã chạy thử toàn bộ pipeline
hai seed × chín cấu hình × hai epoch, gồm chọn cấu hình và xuất dự đoán test.
Kết quả chạy thử này chỉ kiểm tra mã, không dùng để đánh giá chất lượng mô hình.
