# Binary so với multi-class (tự sinh)

2144 ảnh train + val, CV 5 fold (seed fold 2026), seed [42, 43], 300 epoch, cùng descriptor và MLP 256-64-32. Không đọc tập test.

| Cấu hình | C | Ngẫu nhiên | Lớp đông nhất | Accuracy | Balanced acc. | Macro-F1 | κ | MCC | Tham số đầu ra | Giây/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 lớp · softmax + CE | 6 | 16.7% | 23.6% | 76.9% | 74.7% | 0.753 | 0.716 | 0.716 | 198 | 0.28 |
| 6 lớp · 6 sigmoid one-vs-rest + BCE | 6 | 16.7% | 23.6% | 77.2% | 75.2% | 0.756 | 0.720 | 0.720 | 198 | 0.30 |
| Giấy/khác · 1 sigmoid + BCE | 2 | 50.0% | 60.4% | 91.0% | 90.5% | 0.905 | 0.811 | 0.811 | 33 | 0.30 |
| Giấy/khác · 2 softmax + CE | 2 | 50.0% | 60.4% | 91.5% | 91.1% | 0.911 | 0.822 | 0.822 | 66 | 0.38 |
| Trash/khác · 1 sigmoid + BCE | 2 | 50.0% | 94.5% | 96.3% | 76.5% | 0.798 | 0.596 | 0.602 | 33 | 0.55 |
| Trash/khác · BCE có pos_weight | 2 | 50.0% | 94.5% | 95.1% | 80.9% | 0.783 | 0.567 | 0.571 | 33 | 0.33 |
| Giấy/khác · gộp xác suất từ mô hình 6 lớp | 2 | 50.0% | 60.4% | 91.4% | 90.9% | 0.910 | 0.820 | 0.820 | — | — |
| Trash/khác · gộp từ mô hình 6 lớp | 2 | 50.0% | 94.5% | 96.5% | 79.4% | 0.818 | 0.636 | 0.639 | — | — |

## So sánh theo cặp (McNemar chính xác, từng seed)

| So sánh | Chỉ A đúng / chỉ B đúng | p |
| --- | --- | --- |
| fiber_sigmoid_vs_fiber_softmax2 | 52 / 61; 56 / 70 | 0.45; 0.25 |
| fiber_sigmoid_vs_fiber_from_six | 62 / 88; 85 / 77 | 0.041; 0.58 |
| six_softmax_vs_six_ovr | 135 / 140; 124 / 132 | 0.81; 0.66 |
| trash_sigmoid_vs_weighted | 39 / 26; 68 / 30 | 0.14; 0.00016 |

## Lúc khởi tạo (256 ảnh, chế độ eval)

| Cấu hình | Loss ban đầu | ‖∂L/∂z‖ TB | Sai số giải tích vs autograd |
| --- | ---: | ---: | ---: |
| 6 lớp · softmax + CE | 1.448 | 0.856 | 6.0e-08 |
| 6 lớp · 6 sigmoid one-vs-rest + BCE | 3.321 | 1.046 | 0.0e+00 |
| Giấy/khác · 1 sigmoid + BCE | 1.107 | 0.654 | 0.0e+00 |
| Giấy/khác · 2 softmax + CE | 0.322 | 0.367 | 6.0e-08 |
| Trash/khác · 1 sigmoid + BCE | 0.437 | 0.346 | 0.0e+00 |
| Trash/khác · BCE có pos_weight | 0.437 | 0.346 | 0.0e+00 |
