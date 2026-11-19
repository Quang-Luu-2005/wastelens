# Thí nghiệm bộ lọc OpenCV × mô hình

Đề Lab03 yêu cầu phân tích kết quả khi chạy các bộ lọc OpenCV truyền thống cùng toàn bộ pipeline mô hình đa lớp. Thí nghiệm này trả lời ba câu hỏi trên cả ba checkpoint của ứng dụng:

1. Bộ lọc của Module 1 áp lên ảnh sạch thay đổi dự đoán ra sao?
2. Mỗi mô hình chịu được những kiểu suy giảm ảnh nào, đến mức nào?
3. Bộ lọc khôi phục kinh điển có "cứu" được ảnh đã hỏng không?

Kết quả đầy đủ nằm trong [`reports/filter-robustness-20261119/`](../reports/filter-robustness-20261119/results.md), gồm bảng, biểu đồ và dự đoán từng ảnh.

## Thiết lập

| Thành phần | Lựa chọn |
| --- | --- |
| Ảnh | 377 ảnh test TrashNet. Cả ba mô hình đều chưa huấn luyện trên tập này |
| Mô hình | Baseline Lab 2 (MLP pixel 64×64), MLP cải tiến (ensemble 10 MLP descriptor), MobileNetV3 Large (ImageNet pretrained → fine-tune). Checkpoint cố định, không huấn luyện hay chỉnh gì thêm |
| Áp bộ lọc | Lên ảnh gốc 512×384, đúng như Module 1 trong app làm. Mô hình resize sau đó |
| Điều kiện | 76: ảnh gốc, 14 bộ lọc Module 1, 9 kiểu suy giảm × 5 mức, 4 bộ lọc khôi phục trên ảnh sạch (đối chứng), 9 cặp suy giảm → khôi phục |
| Nhiễu ngẫu nhiên | Seed theo chỉ số ảnh, nên mọi mô hình thấy đúng cùng một ảnh hỏng |
| Thống kê | So mỗi điều kiện với chính mô hình đó trên ảnh gốc bằng McNemar chính xác (theo cặp từng ảnh), hiệu chỉnh Holm trên toàn bộ điều kiện của mô hình. Khoảng tin cậy 95% của accuracy trên 377 ảnh rộng khoảng ±4 điểm |

Các kiểu suy giảm và năm mức:

- Nhiễu Gaussian σ: 5, 10, 20, 35, 50
- Muối tiêu: 1 → 15% điểm ảnh
- Gaussian blur σ: 1 → 8
- Giảm tương phản về xám (α): 0,8 → 0,2
- Làm tối (hệ số V): 0,8 → 0,2
- Dịch hue: 5 → 45 đơn vị OpenCV
- Giảm bão hòa (hệ số S): 0,8 → 0
- Ám màu ấm (tăng đỏ, giảm lam): 5 → 30%
- Xoay: 10 → 180°

Mã nguồn: [`wastelens.robustness`](../src/wastelens/robustness.py) và [`scripts/filter_experiment.py`](../scripts/filter_experiment.py). Chạy lại mất khoảng 8 phút trên CPU:

```powershell
.\.venv\Scripts\python.exe scripts/filter_experiment.py --output reports/filter-robustness-20261119
```

## Kết quả tổng quát

| Mô hình | Ảnh gốc | Trung bình 45 điều kiện suy giảm | Giảm tương đối | Số điều kiện khác ảnh gốc có ý nghĩa (Holm) |
| --- | ---: | ---: | ---: | ---: |
| Baseline Lab 2 | 58,6% | 47,8% | 18,5% | 33 / 75 |
| MLP cải tiến | 85,7% | 62,1% | 27,5% | 50 / 75 |
| MobileNetV3 | 94,7% | 84,2% | 11,1% | 38 / 75 |

MLP cải tiến chính xác hơn baseline 27 điểm trên ảnh gốc nhưng mất nhiều nhất khi ảnh bị biến đổi, tính cả theo tỉ lệ tương đối. Ba mô hình hỏng theo ba kiểu khác hẳn nhau ([đường cong theo mức](../reports/filter-robustness-20261119/severity-curves.png), [heatmap](../reports/filter-robustness-20261119/corruption-heatmap.png)).

### MLP cải tiến: bền với nhiễu và blur, rất nhạy với màu và độ sáng

| Suy giảm | Mức nhẹ nhất có thay đổi có ý nghĩa | Mức mạnh nhất |
| --- | --- | --- |
| Gaussian blur | Không mức nào; σ = 8 chỉ −1,6 điểm | 84,1% |
| Muối tiêu | 6% điểm ảnh (−5,0) | 15%: 73,7% (−11,9) |
| Làm tối | V = 0,8 (−18,8) | V = 0,3: 24,4% (−61,3) |
| Dịch hue | 10 đơn vị (−23,6) | 30 đơn vị: 24,9% (−60,7) |
| Ám màu ấm | 5% (−26,3) | 30%: 13,8% (−71,9), gần mức đoán ngẫu nhiên |
| Giảm tương phản | α = 0,8 (−6,9) | α = 0,2: 37,7% (−48,0) |

### MobileNetV3: bền với màu, nhạy với nhiễu xung và blur

- Ám màu 30% chỉ còn −13,5 điểm; làm tối V = 0,3 chỉ −13,3 điểm.
- Ngược lại, muối tiêu 15% làm accuracy rơi từ 94,7% xuống 43,8% (−50,9). Blur σ = 8 còn 58,4% (−36,3).
- Ở nhiễu Gaussian σ = 50, MobileNetV3 (71,9%) còn thấp hơn MLP cải tiến (73,5%), dù trên ảnh gốc cao hơn 9 điểm.

### Baseline Lab 2: không phản ứng với nhiễu và blur, nhạy với màu

Nhiễu Gaussian, muối tiêu và blur không đổi accuracy quá 1,1 điểm ở bất kỳ mức nào. Màu và tương phản thì hại nặng: ám màu 30% còn 16,2% (−42,4); dịch hue 45 còn 27,9% (−30,8).

## Bộ lọc Module 1 trên ảnh sạch

| Bộ lọc | Baseline | MLP cải tiến | MobileNetV3 |
| --- | ---: | ---: | ---: |
| Gaussian σ = 1, median 5×5, bilateral d = 9, preset Nhẹ | 0 → −0,3 | 0 → −0,8 | 0 → −2,1 |
| Lật ngang | −0,8 | −0,5 | +0,3 |
| Preset Vừa (gamma 1,15 · CLAHE 2 · σ 0,8) | −2,1 | −5,6* | −4,5* |
| Preset Mạnh (gamma 1,25 · tương phản 1,15 · CLAHE 3 · σ 1,2) | −7,7* | −19,4* | −8,0* |
| CLAHE clip 4 | −4,5 | −9,0* | −14,1* |
| Gamma 1,5 / 0,67 | −8,8* / −8,8* | −17,0* / −18,8* | −0,8 / −1,9 |
| Gray-world | −24,1* | −27,9* | −2,7 |
| Ảnh xám | −27,9* | −39,3* | −12,7* |

Đơn vị là điểm % so với ảnh gốc; dấu * nghĩa là khác ảnh gốc có ý nghĩa sau hiệu chỉnh Holm. [Biểu đồ](../reports/filter-robustness-20261119/module1-filters.png).

Không bộ lọc nào làm tăng accuracy có ý nghĩa trên ảnh TrashNet sạch. Các bộ lọc làm mượt nhẹ gần như vô hại. Mọi phép đổi màu, tương phản hoặc độ sáng đều làm giảm accuracy, và mức giảm lớn dần từ MobileNetV3 lên baseline rồi lên MLP cải tiến. Preset mạnh hơn thì kém hơn, đúng như ghi chú trong app: mức xử lý mạnh hơn không đảm bảo phân loại tốt hơn.

## Bộ lọc khôi phục có cứu được ảnh hỏng không

Bảng dưới so số ảnh đúng khi chỉ bị suy giảm với số ảnh đúng sau bộ lọc khôi phục, theo cặp từng ảnh, McNemar với hiệu chỉnh Holm trên 27 phép so ([biểu đồ](../reports/filter-robustness-20261119/rescue.png)).

| Suy giảm → bộ lọc | Baseline | MLP cải tiến | MobileNetV3 |
| --- | --- | --- | --- |
| Muối tiêu 6% → median 3×3 | 219 → 220 | 304 → 322* | 230 → 356* |
| Nhiễu Gaussian σ 35 → bilateral d = 9 | 219 → 220 | 290 → 293 | 303 → 316 |
| Làm tối V 0,3 → gamma 2,0 | 195 → 160* | 92 → 121* | 307 → 332* |
| Làm tối V 0,3 → CLAHE 3 | 195 → 127* | 92 → 97 | 307 → 295 |
| Tương phản α 0,3 → CLAHE 3 | 137 → 138 | 148 → 156 | 298 → 311 |
| Ám màu 20% → gray-world | 75 → 127* | 74 → 206* | 321 → 346* |

Ảnh gốc lần lượt đúng 221, 323 và 357 ảnh.

- **Median trị muối tiêu gần như hoàn toàn.** MobileNetV3 lấy lại 126 ảnh, từ 61,0% lên 94,4%, gần bằng ảnh gốc 94,7%. MLP cải tiến về 85,4%. Đây là cặp "bệnh và thuốc" thành công nhất.
- **Gray-world cứu được phần lớn ám màu, nhưng không vượt quá chính nó trên ảnh sạch.** MLP cải tiến tăng từ 19,6% lên 54,6% (+132 ảnh). Tuy vậy, gray-world áp lên ảnh sạch đã kéo mô hình này xuống 57,8%. Bộ lọc đưa ảnh về "màu trung tính" chứ không về màu của buổi chụp TrashNet mà mô hình đã học.
- **Gamma giúp với ảnh tối ở hai mô hình mạnh, nhưng hại baseline.** CLAHE sau làm tối thì không giúp, và còn làm baseline mất thêm 68 ảnh.
- **Làm mượt sau nhiễu Gaussian không có hiệu quả có ý nghĩa** ở mô hình nào; MobileNetV3 chỉ nhích từ 303 lên 316 ảnh.

Kết luận này khớp với tài liệu: khôi phục ảnh chỉ áp ở bước suy luận hiếm khi đưa accuracy về mức của ảnh sạch, và cân bằng trắng hiệu quả nhất khi áp cho cả dữ liệu train lẫn ảnh khi chạy ([Pei et al. 2018](https://arxiv.org/abs/1810.05552); [Afifi & Brown 2019](https://arxiv.org/abs/1912.06960); [Hendrycks & Dietterich 2019](https://arxiv.org/abs/1903.12261)).

## Giải thích cơ chế

- **Vì sao MLP cải tiến nhạy với màu.** 1.008 trong 2.772 giá trị đầu vào là histogram HSV và Lab với 8 bin mỗi kênh, trên lưới không gian 1/2/4. Một bin hue rộng 180/8 = 22,5 đơn vị OpenCV. Vì vậy dịch hue 10–20 đơn vị đã đẩy phần lớn điểm ảnh sang bin bên cạnh, và vector đặc trưng đổi hẳn. Làm tối và ám màu dịch kênh V, L, a, b theo cùng cách. Năm mạng HSV-only của ensemble chỉ nhìn thấy màu. Phân tích out-of-fold trước đó cũng cho thấy chỉ riêng màu nền đã đoán đúng lớp 54,7% ảnh. Màu là tín hiệu mạnh của TrashNet, nhưng gắn chặt với điều kiện chụp.
- **Vì sao nó bền với blur và nhiễu.** Descriptor được tính trên ảnh thu về 64×64, tức nhỏ khoảng 8 lần mỗi chiều. Blur σ = 8 ở ảnh gốc chỉ còn khoảng 1 điểm ảnh ở 64×64. Nhiễu độc lập theo điểm ảnh cũng bị trung bình hóa khi thu nhỏ. HOG chuẩn hóa theo block nên gần như không đổi khi độ sáng thay đổi tuyến tính, nhưng phần histogram màu thì có đổi.
- **Vì sao MobileNetV3 ngược lại.** Đầu vào 224×224 giữ nhiều chi tiết tần số cao hơn: blur σ = 8 vẫn còn khoảng 3,5 điểm ảnh. CNN dựa nhiều vào kết cấu cục bộ nên nhạy với blur và nhiễu xung. Ngược lại, đặc trưng học từ ImageNet ít phụ thuộc vào màu tuyệt đối hơn histogram cố định.
- **Một yếu tố gây nhiễu cần ghi rõ.** Vì bộ lọc được áp ở 512×384 rồi mỗi mô hình resize khác nhau, "cùng mức blur" không có nghĩa là cùng mức blur trên đầu vào của mô hình. Sự bền với blur của hai MLP một phần đến từ độ phân giải đầu vào thấp, không hẳn là ưu điểm của đặc trưng.

## Hệ quả cho ứng dụng và báo cáo

1. Với ảnh giống TrashNet, Module 1 là công cụ minh họa và thử nghiệm, không phải bước tiền xử lý làm tăng accuracy. Mặc định nên để "Gốc".
2. Với MLP cải tiến, điều kiện ánh sáng và cân bằng trắng của webcam là rủi ro lớn nhất: ám màu chỉ 5% đã mất 26 điểm. Nhiễu cảm biến hay ảnh hơi mờ ít đáng lo hơn.
3. Muốn cải thiện thật thì phải đưa biến đổi màu vào lúc train (augmentation màu, cân bằng trắng cho cả train và test, hoặc dữ liệu tự chụp), không chỉ lọc ảnh lúc chạy.
4. Trong app, khi bật Module 1, kết quả trên ảnh gốc và ảnh đã xử lý được hiển thị cạnh nhau. Đó là cách minh họa trực tiếp các con số ở trên.

## Hạn chế

- Tập test đã được xem trong các vòng chọn mô hình trước đó. Thí nghiệm này không chọn hay chỉnh gì nên không làm tăng thiên lệch, nhưng mức accuracy tuyệt đối vẫn lạc quan.
- 377 ảnh, trong đó 20 ảnh trash. So sánh theo cặp phát hiện được thay đổi vài điểm, nhưng recall theo lớp còn rất bất định.
- Suy giảm là giả lập. Ánh sáng thật, nền lộn xộn và vật bị che chưa được đo; cần bộ ảnh webcam tự chụp.
- Mức độ được chọn theo kinh nghiệm cho ảnh 512×384. Thang đo mức không so sánh được giữa các kiểu suy giảm khác nhau.
