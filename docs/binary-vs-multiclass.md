# Binary (Lab02) so với multi-class (Lab03)

Đề Lab03 yêu cầu phân tích khác biệt giữa bài toán nhị phân của Lab02 và bài toán đa lớp của Lab03 ở ba mặt: hàm loss, cách tính gradient và cập nhật trọng số, và độ phức tạp tính toán. Tài liệu gồm hai phần. Phần lý thuyết có công thức kiểm chứng được bằng code. Phần thực nghiệm giữ nguyên dữ liệu, đặc trưng, kiến trúc và cách huấn luyện, chỉ đổi số lớp, lớp đầu ra và hàm loss.

## 1. Mô hình hai bài lab

| | Lab02 (Horse or Human) | Lab03 (TrashNet) |
| --- | --- | --- |
| Số lớp | 2 | 6 |
| Đầu ra | 1 logit z → sigmoid | 6 logit z₁…z₆ → softmax |
| Loss | Binary cross-entropy | Categorical cross-entropy |
| Lớp đầu ra của MLP pixel 12.288 → 128 → 32 | 32 → 1: 33 tham số | 32 → 6: 198 tham số |
| Tổng tham số MLP pixel | 1.577.153 | 1.577.318 (+0,010%) |
| Tổng tham số một MLP descriptor 2.772 → 256 → 64 → 32 | 729.153 nếu là nhị phân | 729.318 (+0,023%) |
| Dự đoán | p ≥ 0,5 | argmax |

Phần thân mạng (các tầng ẩn) không đổi. Chỉ lớp cuối và hàm loss thay đổi, cùng cách đọc ra nhãn.

## 2. Hàm loss

**Nhị phân.** Với logit z và nhãn y ∈ {0, 1}:

  p = σ(z) = 1 / (1 + e^(−z)),  L = −[y·log p + (1 − y)·log(1 − p)]

**Đa lớp.** Với vector logit z ∈ ℝ^C và nhãn y ∈ {1, …, C}:

  p_k = e^(z_k) / Σ_j e^(z_j),  L = −log p_y = −z_y + log Σ_j e^(z_j)

Cross-entropy đa lớp là tổng quát hóa trực tiếp của BCE. Với C = 2, softmax của (z₀, z₁) bằng σ(z₁ − z₀), nên mô hình softmax 2 logit và mô hình sigmoid 1 logit biểu diễn cùng một họ hàm. Khác biệt duy nhất là softmax 2 logit có một tham số dư (Goodfellow et al., *Deep Learning*, mục 6.2.2.3).

**Ổn định số.** Không tính log(σ(z)) hay log Σ e^(z_j) trực tiếp, vì e^z tràn số khi z lớn. PyTorch dùng các dạng tương đương sau, và code dự án gọi đúng hai hàm này (`BCEWithLogitsLoss`, `CrossEntropyLoss`):

- BCE: L = max(z, 0) − z·y + log(1 + e^(−|z|))
- Cross-entropy: L = −z_y + m + log Σ_j e^(z_j − m), với m = max_j z_j

**Giá trị lúc khởi tạo.** Khi mạng chưa học, xác suất xấp xỉ đều: loss nhị phân ≈ ln 2 = 0,693, loss 6 lớp ≈ ln 6 = 1,792. Hai đường loss vì vậy không so trực tiếp được. Phải chia cho ln C, khi đó 1,0 nghĩa là ngang đoán đều.

## 3. Gradient và cập nhật trọng số

**Gradient theo logit.** Cả hai trường hợp có cùng dạng "dự đoán trừ nhãn" (Bishop, *PRML*, công thức 4.91 và 4.109):

- Nhị phân: ∂L/∂z = σ(z) − y. Đạo hàm dùng σ′(z) = σ(z)(1 − σ(z)).
- Đa lớp: ∂L/∂z_k = p_k − 1[k = y], tức vector p − onehot(y).

Với đa lớp, đạo hàm của softmax là ma trận Jacobian ∂p_i/∂z_j = p_i(δ_ij − p_j), viết gọn là diag(p) − p·pᵀ. Ghép với −log p_y cho đúng p − onehot(y). Các thành phần của vector này cộng lại bằng 0: logit của lớp đúng bị kéo lên đúng bằng tổng lượng các logit lớp sai bị đẩy xuống.

**Lan truyền ngược vào tầng ẩn cuối.** Gọi h ∈ ℝ³² là tầng ẩn cuối:

- Nhị phân, trọng số w ∈ ℝ³²: δ_h = (p − y)·w. Một số vô hướng nhân một vector.
- Đa lớp, trọng số W ∈ ℝ^(C×32): δ_h = Wᵀ(p − onehot(y)). Đây là tổ hợp có trọng số của C hàng: mỗi lớp sai "đẩy" h theo hướng hàng của nó, tỉ lệ với xác suất nó đang nhận.

Từ đây trở xuống (các tầng ẩn, LayerNorm, ReLU), backprop giống hệt hai bài, chỉ có tín hiệu lỗi ở đỉnh khác đi.

**Cập nhật lớp đầu ra.** Với SGD, η là learning rate:

- Nhị phân: w ← w − η(p − y)·h
- Đa lớp: W ← W − η(p − onehot(y))·hᵀ. Hàng của lớp đúng được kéo về phía h với hệ số 1 − p_y; hàng của mỗi lớp sai bị đẩy ra xa với hệ số p_k.

Cả hai bài dùng Adam. Adam chuẩn hóa độ lớn gradient theo từng tham số, nhưng hướng cập nhật vẫn do các gradient trên quyết định.

**Độ lớn gradient lúc khởi tạo.** Với xác suất đều: nhị phân |σ(z) − y| = 0,5; 6 lớp ‖p − onehot(y)‖ = √((5/6)² + 5·(1/6)²) = √(5/6) ≈ 0,913.

**Hai cách tách đa lớp khác.** Softmax không phải cách duy nhất:

- **One-vs-rest:** C sigmoid độc lập, mỗi đầu ra học bài "lớp k hay không" bằng BCE, gradient σ(z_k) − y_k riêng cho từng đầu ra (Bishop, mục 4.1.2). Các xác suất không buộc cộng lại bằng 1, nên có thể mâu thuẫn nhau.
- **Gộp:** từ xác suất 6 lớp có thể trả lời bất kỳ câu hỏi nhị phân nào bằng cách cộng xác suất theo nhóm.

Cả hai đều được đo trong phần thực nghiệm.

## 4. Độ phức tạp

**Tính toán.** Lớp đầu ra cần H·C phép nhân-cộng mỗi mẫu (H = 32): từ 32 lên 192. Softmax thêm C phép mũ. So với tầng đầu của MLP descriptor (2.772 × 256 = 709.632 phép nhân-cộng), phần tăng thêm khoảng 0,02%. Thời gian huấn luyện và suy luận gần như không đổi; số đo nằm ở phần 5.

**Thống kê.** Phần thực sự tăng theo C là độ khó của bài học:

- Mức đoán ngẫu nhiên giảm từ 1/2 xuống 1/6.
- Cùng số ảnh, mỗi lớp có ít mẫu hơn (lớp trash chỉ 117/2.144 ảnh).
- Số mẫu cần để học tăng theo số lớp: với bộ phân loại tuyến tính đa lớp, chiều Natarajan nằm giữa (C − 1)(d − 1) và C·d (Shalev-Shwartz & Ben-David, *Understanding Machine Learning*, chương 29).

**Đánh giá.** Bài nhị phân có một ngưỡng, đường ROC và precision/recall của lớp dương. Bài đa lớp cần ma trận nhầm lẫn C × C và các chỉ số trung bình theo lớp. Accuracy không so được giữa hai bài có C khác nhau, hay có mức mất cân bằng khác nhau. Vì vậy tài liệu dùng thêm balanced accuracy, Cohen's κ và MCC: các chỉ số này trừ phần đúng do may mắn và bằng 0 khi chỉ đoán theo lớp đông nhất.

## 5. Thực nghiệm

**Thiết kế.** Không so trực tiếp Horse-or-Human với TrashNet, vì hai bộ dữ liệu khác nhau cả về ảnh lẫn độ khó. Thay vào đó, mọi thứ được giữ nguyên trên TrashNet:

- cùng 2.144 ảnh train + val và cùng descriptor HOG + HSV + Lab;
- cùng MLP 256-64-32 với LayerNorm và dropout 0,25, cùng Adam lr 0,001, batch 64, 300 epoch, lật ngang;
- cùng 5 fold phân tầng, trùng với repeat 0 trong CV của mô hình triển khai; 2 seed (42, 43);
- không đọc tập test.

Chỉ thay đổi ánh xạ nhãn, số đơn vị đầu ra và hàm loss. Có ba bài:

- **6 lớp:** bài của Lab03.
- **Giấy/khác:** {cardboard, paper} so với phần còn lại, tỉ lệ 848 : 1.296. Đây là bài nhị phân chính, giữ đủ mọi ảnh và cùng fold.
- **Trash/khác:** 117 : 2.027, dùng để minh họa mất cân bằng lớp.

Code: [`wastelens.binary_multiclass`](../src/wastelens/binary_multiclass.py) và [`scripts/binary_vs_multiclass.py`](../scripts/binary_vs_multiclass.py). Thời gian chạy 21,6 phút trên CPU cho 60 lần fit. Bảng, biểu đồ và JSON nằm trong [`reports/binary-vs-multiclass-20261119/`](../reports/binary-vs-multiclass-20261119/results.md).

### Kết quả out-of-fold (trung bình 2 seed)

| Cấu hình | C | Luôn đoán lớp đông nhất | Accuracy | Balanced acc. | Cohen κ | MCC | Tham số lớp ra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 lớp · softmax + CE | 6 | 23,6% | 76,9% | 74,7% | 0,716 | 0,716 | 198 |
| 6 lớp · 6 sigmoid one-vs-rest + BCE | 6 | 23,6% | 77,2% | 75,2% | 0,720 | 0,720 | 198 |
| Giấy/khác · 1 sigmoid + BCE | 2 | 60,4% | 91,0% | 90,5% | 0,811 | 0,811 | 33 |
| Giấy/khác · 2 softmax + CE | 2 | 60,4% | 91,5% | 91,1% | 0,822 | 0,822 | 66 |
| Giấy/khác · gộp xác suất từ mô hình 6 lớp | 2 | 60,4% | 91,4% | 90,9% | 0,820 | 0,820 | — |
| Trash/khác · 1 sigmoid + BCE | 2 | 94,5% | 96,3% | 76,5% | 0,596 | 0,602 | 33 |
| Trash/khác · BCE có `pos_weight` | 2 | 94,5% | 95,1% | 80,9% | 0,567 | 0,571 | 33 |
| Trash/khác · gộp từ mô hình 6 lớp | 2 | 94,5% | 96,5% | 79,4% | 0,636 | 0,639 | — |

Có thể so cấu hình 6 lớp ở đây (76,9%) với accuracy out-of-fold 78,8% của ensemble triển khai: ở đây chỉ có một mạng và chỉ một repeat. Biểu đồ: [chỉ số](../reports/binary-vs-multiclass-20261119/metrics.png), [đường huấn luyện](../reports/binary-vs-multiclass-20261119/training-curves.png), [ma trận nhầm lẫn](../reports/binary-vs-multiclass-20261119/confusion.png).

### Nhận xét

1. **Accuracy phóng đại khoảng cách giữa hai bài.** Cùng đặc trưng và cùng mạng, accuracy tăng từ 76,9% (6 lớp) lên 91,0% (nhị phân), chênh 14 điểm. Nhưng hai bài có mức tham chiếu khác nhau: luôn đoán lớp đông nhất đã đạt 23,6% ở bài 6 lớp và 60,4% ở bài nhị phân. Sau khi trừ phần đúng do may mắn, κ chỉ tăng từ 0,716 lên 0,811. Cả hai mô hình đều giải thích được phần lớn phần "không ngẫu nhiên" của bài toán; bài nhị phân dễ hơn ít hơn nhiều so với con số accuracy gợi ý.
2. **Sigmoid 1 logit và softmax 2 logit cho kết quả như nhau,** đúng như lý thuyết về sự tương đương (91,0% so với 91,5%; McNemar p = 0,45 và 0,25 ở hai seed). Lớp đầu ra có gấp đôi tham số (66 so với 33), nhưng phần tham số dư không thêm khả năng biểu diễn.
3. **Mô hình 6 lớp trả lời câu hỏi nhị phân tốt ngang mô hình nhị phân chuyên biệt.** Chỉ cần cộng xác suất cardboard + paper, nó đạt 91,4% so với 91,0%. McNemar: p = 0,041 nghiêng về mô hình 6 lớp ở seed 42, p = 0,58 ở seed 43; với hai seed và hai phép thử, không thể kết luận bên nào hơn. Với trash, mô hình 6 lớp còn cao nhất (κ 0,636). Huấn luyện với nhãn chi tiết không làm mất thông tin cho câu hỏi thô; nó còn buộc mạng phân biệt glass, metal, plastic bên trong lớp "khác".
4. **Softmax và one-vs-rest cho bài 6 lớp ngang nhau:** 76,9% so với 77,2%, p = 0,81 và 0,66. Khác biệt thực tế là softmax cho xác suất cộng lại bằng 1. Sáu sigmoid độc lập có thể cùng cao hoặc cùng thấp, nên không dùng trực tiếp làm confidence được. Đó là lý do mô hình triển khai dùng softmax rồi hiệu chỉnh nhiệt độ.
5. **Mất cân bằng lớp là nơi accuracy đánh lừa nhiều nhất.** Ở bài trash/khác, luôn đoán "khác" đã đạt 94,5%. Mô hình đạt 96,3% nhưng balanced accuracy chỉ 76,5%: ở seed 42 nó chỉ tìm được 64/117 ảnh trash (55%). `pos_weight` = 2.027/117 nâng balanced accuracy lên 80,9% nhưng làm accuracy giảm (95,1%; giảm có ý nghĩa ở seed 43, p = 0,00016), và κ cũng giảm. Đổi trọng số chỉ dời điểm cân bằng giữa bỏ sót và báo nhầm, không làm mô hình phân biệt tốt hơn.
6. **Đường loss và gradient.**
   - Sau khi chia cho ln C, loss softmax của hai bài cùng bắt đầu gần 1 (gần mức đoán đều) và giảm theo cùng dạng.
   - Loss 6 lớp giảm chậm hơn và dừng cao hơn: bài có nhiều ranh giới cần học hơn.
   - Chuẩn ‖∂L/∂z‖ trung bình mỗi mẫu đi theo loss: ở 20 epoch cuối, gradient của bài 6 lớp vẫn lớn hơn 3,4 lần (so với softmax 2 logit) đến 6,6 lần (so với sigmoid 1 logit), vì vẫn còn nhiều mẫu bị phân loại chưa chắc.
   - Loss của one-vs-rest được tính trung bình trên từng đầu ra (BCE trung bình trên N × 6 phần tử), nên không đặt chung thang với cross-entropy.
7. **Gradient giải tích được kiểm chứng.** Trên 256 ảnh lúc khởi tạo, p − onehot(y) và σ(z) − y khớp với `logits.grad` của PyTorch, sai số tối đa 6·10⁻⁸ (sai số làm tròn float32). Loss ban đầu đo được không đúng bằng ln C (ví dụ 1,448 thay vì 1,792 với 6 lớp). Lý do: với khởi tạo Xavier và đầu vào đã chuẩn hóa, logit ban đầu chưa bằng 0. Giá trị ln C là kỳ vọng khi mạng dự đoán đều.
8. **Độ phức tạp tính toán không khác biệt thực tế.** Thời gian mỗi epoch dao động 0,28–0,55 giây giữa các cấu hình. Biến động này do tải CPU khi 5 tiến trình chạy song song, không theo số lớp: bài 6 lớp softmax là 0,28 giây, thấp nhất. Lớp đầu ra thêm 165 tham số trên 729.153 (+0,023%).

### Kết luận cho báo cáo

Chuyển từ nhị phân sang đa lớp chỉ thay đổi lớp đầu ra (32 → 1 thành 32 → 6) và hàm loss (sigmoid + BCE thành softmax + CE). Gradient tại logit giữ nguyên dạng "dự đoán trừ nhãn". Chi phí tính toán tăng không đáng kể. Thứ thực sự thay đổi là độ khó thống kê và cách đánh giá: mức ngẫu nhiên thấp hơn, ít mẫu hơn cho mỗi lớp, cần ma trận nhầm lẫn C × C và các chỉ số trừ phần may mắn (κ, MCC, balanced accuracy) thay cho accuracy thô. Thực nghiệm cho thấy mô hình 6 lớp vẫn trả lời tốt các câu hỏi nhị phân suy ra từ nó, nên một mô hình đa lớp duy nhất là lựa chọn hợp lý cho ứng dụng.

## Tài liệu tham khảo

- Goodfellow, Bengio & Courville, *Deep Learning* (2016), mục 6.2.2.2 (đơn vị sigmoid) và 6.2.2.3 (đơn vị softmax): https://www.deeplearningbook.org/contents/mlp.html
- Bishop, *Pattern Recognition and Machine Learning* (2006): mục 4.1.2 (đa lớp, one-vs-rest), 4.3.2 (logistic regression, công thức 4.91), 4.3.4 (softmax đa lớp, công thức 4.109).
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (2014), chương 29: tính học được đa lớp và chiều Natarajan.
- CS231n, *Linear classification: softmax classifier*: https://cs231n.github.io/linear-classify/
