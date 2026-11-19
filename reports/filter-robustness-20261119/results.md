# Thí nghiệm filter × model (tự sinh)

377 ảnh test, ba checkpoint cố định. Ô bảng: accuracy % (thay đổi so với ảnh gốc, điểm %). Dấu * = khác ảnh gốc có ý nghĩa theo McNemar chính xác, hiệu chỉnh Holm trên mọi điều kiện của mô hình đó.

## Tóm tắt

| Mô hình | Ảnh gốc | TB 45 điều kiện suy giảm | Giảm tương đối | Số điều kiện khác có ý nghĩa |
| --- | ---: | ---: | ---: | ---: |
| Baseline Lab 2 | 58.6% | 47.8% | 18.5% | 33 |
| MLP cải tiến | 85.7% | 62.1% | 27.5% | 50 |
| MobileNetV3 | 94.7% | 84.2% | 11.1% | 38 |

## Bộ lọc Module 1 trên ảnh sạch

| Điều kiện | Baseline Lab 2 | MLP cải tiến | MobileNetV3 |
| --- | ---: | ---: | ---: |
| Preset Nhẹ (Gaussian σ=0,4) | 58.6 (+0.0) | 85.4 (-0.3) | 94.7 (+0.0) |
| Preset Vừa (gamma 1,15 · CLAHE 2 · Gaussian σ=0,8) | 56.5 (-2.1) | 80.1 (-5.6)* | 90.2 (-4.5)* |
| Preset Mạnh (gamma 1,25 · tương phản 1,15 · CLAHE 3 · Gaussian σ=1,2) | 50.9 (-7.7)* | 66.3 (-19.4)* | 86.7 (-8.0)* |
| Gaussian σ=1 | 58.6 (+0.0) | 85.4 (-0.3) | 94.7 (+0.0) |
| Median 5×5 | 58.6 (+0.0) | 84.9 (-0.8) | 93.1 (-1.6) |
| Bilateral d=9, σ=75 | 58.4 (-0.3) | 85.7 (+0.0) | 92.6 (-2.1) |
| CLAHE clip 2 | 55.7 (-2.9) | 78.2 (-7.4)* | 88.9 (-5.8)* |
| CLAHE clip 4 | 54.1 (-4.5) | 76.7 (-9.0)* | 80.6 (-14.1)* |
| Gamma 1,5 (sáng hơn) | 49.9 (-8.8)* | 68.7 (-17.0)* | 93.9 (-0.8) |
| Gamma 0,67 (tối hơn) | 49.9 (-8.8)* | 66.8 (-18.8)* | 92.8 (-1.9) |
| Cân bằng trắng gray-world | 34.5 (-24.1)* | 57.8 (-27.9)* | 92.0 (-2.7) |
| Ảnh xám | 30.8 (-27.9)* | 46.4 (-39.3)* | 82.0 (-12.7)* |
| Lật ngang | 57.8 (-0.8) | 85.1 (-0.5) | 95.0 (+0.3) |
| Crop giữ 80% trung tâm | 54.1 (-4.5) | 78.5 (-7.2)* | 93.6 (-1.1) |

## Suy giảm 5 mức

| Điều kiện | Baseline Lab 2 | MLP cải tiến | MobileNetV3 |
| --- | ---: | ---: | ---: |
| Nhiễu Gaussian σ = 5 | 58.6 (+0.0) | 83.3 (-2.4) | 92.6 (-2.1) |
| Nhiễu Gaussian σ = 10 | 58.4 (-0.3) | 83.0 (-2.7) | 90.7 (-4.0) |
| Nhiễu Gaussian σ = 20 | 58.6 (+0.0) | 81.2 (-4.5) | 87.5 (-7.2)* |
| Nhiễu Gaussian σ = 35 | 58.1 (-0.5) | 76.9 (-8.8)* | 80.4 (-14.3)* |
| Nhiễu Gaussian σ = 50 | 58.4 (-0.3) | 73.5 (-12.2)* | 71.9 (-22.8)* |
| Nhiễu muối tiêu (tỉ lệ) = 0.01 | 58.9 (+0.3) | 83.8 (-1.9) | 88.1 (-6.6)* |
| Nhiễu muối tiêu (tỉ lệ) = 0.03 | 58.1 (-0.5) | 82.5 (-3.2) | 78.8 (-15.9)* |
| Nhiễu muối tiêu (tỉ lệ) = 0.06 | 58.1 (-0.5) | 80.6 (-5.0)* | 61.0 (-33.7)* |
| Nhiễu muối tiêu (tỉ lệ) = 0.1 | 57.8 (-0.8) | 77.2 (-8.5)* | 49.9 (-44.8)* |
| Nhiễu muối tiêu (tỉ lệ) = 0.15 | 57.6 (-1.1) | 73.7 (-11.9)* | 43.8 (-50.9)* |
| Gaussian blur σ = 1 | 58.6 (+0.0) | 85.4 (-0.3) | 94.7 (+0.0) |
| Gaussian blur σ = 2 | 58.4 (-0.3) | 85.1 (-0.5) | 92.0 (-2.7) |
| Gaussian blur σ = 3 | 58.9 (+0.3) | 85.4 (-0.3) | 84.4 (-10.3)* |
| Gaussian blur σ = 5 | 59.2 (+0.5) | 84.4 (-1.3) | 72.9 (-21.8)* |
| Gaussian blur σ = 8 | 58.9 (+0.3) | 84.1 (-1.6) | 58.4 (-36.3)* |
| Giảm tương phản α = 0.8 | 54.9 (-3.7) | 78.8 (-6.9)* | 95.0 (+0.3) |
| Giảm tương phản α = 0.6 | 46.9 (-11.7)* | 60.2 (-25.5)* | 92.6 (-2.1) |
| Giảm tương phản α = 0.45 | 39.8 (-18.8)* | 50.7 (-35.0)* | 88.3 (-6.4)* |
| Giảm tương phản α = 0.3 | 36.3 (-22.3)* | 39.3 (-46.4)* | 79.0 (-15.6)* |
| Giảm tương phản α = 0.2 | 33.4 (-25.2)* | 37.7 (-48.0)* | 63.1 (-31.6)* |
| Làm tối (hệ số V) = 0.8 | 57.0 (-1.6) | 66.8 (-18.8)* | 93.9 (-0.8) |
| Làm tối (hệ số V) = 0.6 | 57.0 (-1.6) | 37.4 (-48.3)* | 93.4 (-1.3) |
| Làm tối (hệ số V) = 0.45 | 56.0 (-2.7) | 26.3 (-59.4)* | 87.8 (-6.9)* |
| Làm tối (hệ số V) = 0.3 | 51.7 (-6.9) | 24.4 (-61.3)* | 81.4 (-13.3)* |
| Làm tối (hệ số V) = 0.2 | 43.2 (-15.4)* | 32.4 (-53.3)* | 66.0 (-28.6)* |
| Dịch hue (đơn vị OpenCV) = 5 | 55.7 (-2.9) | 81.2 (-4.5) | 93.6 (-1.1) |
| Dịch hue (đơn vị OpenCV) = 10 | 53.1 (-5.6)* | 62.1 (-23.6)* | 92.6 (-2.1) |
| Dịch hue (đơn vị OpenCV) = 20 | 45.6 (-13.0)* | 44.0 (-41.6)* | 89.4 (-5.3)* |
| Dịch hue (đơn vị OpenCV) = 30 | 38.2 (-20.4)* | 24.9 (-60.7)* | 87.5 (-7.2)* |
| Dịch hue (đơn vị OpenCV) = 45 | 27.9 (-30.8)* | 43.0 (-42.7)* | 85.9 (-8.8)* |
| Giảm bão hòa (hệ số S) = 0.8 | 57.6 (-1.1) | 83.0 (-2.7) | 94.4 (-0.3) |
| Giảm bão hòa (hệ số S) = 0.6 | 50.9 (-7.7)* | 73.2 (-12.5)* | 94.7 (+0.0) |
| Giảm bão hòa (hệ số S) = 0.4 | 41.4 (-17.2)* | 66.8 (-18.8)* | 92.0 (-2.7) |
| Giảm bão hòa (hệ số S) = 0.2 | 36.9 (-21.8)* | 63.9 (-21.8)* | 88.1 (-6.6)* |
| Giảm bão hòa (hệ số S) = 0 | 31.6 (-27.1)* | 50.1 (-35.5)* | 80.1 (-14.6)* |
| Ám màu ấm (cường độ) = 0.05 | 40.6 (-18.0)* | 59.4 (-26.3)* | 93.4 (-1.3) |
| Ám màu ấm (cường độ) = 0.1 | 26.3 (-32.4)* | 41.9 (-43.8)* | 90.2 (-4.5)* |
| Ám màu ấm (cường độ) = 0.15 | 21.8 (-36.9)* | 28.4 (-57.3)* | 87.3 (-7.4)* |
| Ám màu ấm (cường độ) = 0.2 | 19.9 (-38.7)* | 19.6 (-66.0)* | 85.1 (-9.5)* |
| Ám màu ấm (cường độ) = 0.3 | 16.2 (-42.4)* | 13.8 (-71.9)* | 81.2 (-13.5)* |
| Xoay (độ) = 10 | 55.7 (-2.9) | 83.3 (-2.4) | 94.7 (+0.0) |
| Xoay (độ) = 20 | 49.3 (-9.3)* | 82.8 (-2.9) | 94.2 (-0.5) |
| Xoay (độ) = 45 | 44.0 (-14.6)* | 71.1 (-14.6)* | 90.7 (-4.0) |
| Xoay (độ) = 90 | 42.2 (-16.4)* | 61.0 (-24.7)* | 90.7 (-4.0) |
| Xoay (độ) = 180 | 43.8 (-14.9)* | 68.2 (-17.5)* | 95.0 (+0.3) |

## Đối chứng: bộ lọc khôi phục trên ảnh sạch

| Điều kiện | Baseline Lab 2 | MLP cải tiến | MobileNetV3 |
| --- | ---: | ---: | ---: |
| Ảnh sạch → Median 3×3 | 58.4 (-0.3) | 85.1 (-0.5) | 94.4 (-0.3) |
| Ảnh sạch → Median 5×5 | 58.6 (+0.0) | 84.9 (-0.8) | 93.1 (-1.6) |
| Ảnh sạch → Gaussian σ=1 | 58.6 (+0.0) | 85.4 (-0.3) | 94.7 (+0.0) |
| Ảnh sạch → Bilateral d=9 | 58.4 (-0.3) | 85.7 (+0.0) | 92.6 (-2.1) |
| Ảnh sạch → Gamma 2,0 (làm sáng) | 44.3 (-14.3)* | 55.4 (-30.2)* | 92.6 (-2.1) |
| Ảnh sạch → CLAHE clip 3 | 54.1 (-4.5) | 79.8 (-5.8)* | 85.1 (-9.5)* |
| Ảnh sạch → Gray-world | 34.5 (-24.1)* | 57.8 (-27.9)* | 92.0 (-2.7) |

## Khôi phục sau suy giảm

| Điều kiện | Baseline Lab 2 | MLP cải tiến | MobileNetV3 |
| --- | ---: | ---: | ---: |
| Nhiễu muối tiêu (tỉ lệ) = 0.06 → Median 3×3 | 58.4 (-0.3) | 85.4 (-0.3) | 94.4 (-0.3) |
| Nhiễu muối tiêu (tỉ lệ) = 0.06 → Median 5×5 | 58.6 (+0.0) | 84.9 (-0.8) | 92.6 (-2.1) |
| Nhiễu Gaussian σ = 35 → Gaussian σ=1 | 58.1 (-0.5) | 76.9 (-8.8)* | 82.5 (-12.2)* |
| Nhiễu Gaussian σ = 35 → Bilateral d=9 | 58.4 (-0.3) | 77.7 (-8.0)* | 83.8 (-10.9)* |
| Nhiễu Gaussian σ = 35 → Median 5×5 | 58.6 (+0.0) | 77.5 (-8.2)* | 81.2 (-13.5)* |
| Làm tối (hệ số V) = 0.3 → Gamma 2,0 (làm sáng) | 42.4 (-16.2)* | 32.1 (-53.6)* | 88.1 (-6.6)* |
| Làm tối (hệ số V) = 0.3 → CLAHE clip 3 | 33.7 (-24.9)* | 25.7 (-59.9)* | 78.2 (-16.4)* |
| Giảm tương phản α = 0.3 → CLAHE clip 3 | 36.6 (-22.0)* | 41.4 (-44.3)* | 82.5 (-12.2)* |
| Ám màu ấm (cường độ) = 0.2 → Gray-world | 33.7 (-24.9)* | 54.6 (-31.0)* | 91.8 (-2.9) |

## Khôi phục so với ảnh hỏng (cùng mô hình, theo cặp)

Số ảnh đúng khi chỉ bị suy giảm → sau bộ lọc khôi phục; mất/được = ảnh chỉ đúng ở một bên. p theo McNemar chính xác, hiệu chỉnh Holm trên toàn bộ bảng.

| Mô hình | Suy giảm → bộ lọc | Đúng | Thay đổi | Mất / được | p (Holm) |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline Lab 2 | salt_pepper_3 → Median 3×3 | 219 → 220 | +1 | 7 / 8 | 1 |
| Baseline Lab 2 | salt_pepper_3 → Median 5×5 | 219 → 221 | +2 | 7 / 9 | 1 |
| Baseline Lab 2 | gaussian_noise_4 → Gaussian σ=1 | 219 → 219 | +0 | 0 / 0 | 1 |
| Baseline Lab 2 | gaussian_noise_4 → Bilateral d=9 | 219 → 220 | +1 | 0 / 1 | 1 |
| Baseline Lab 2 | gaussian_noise_4 → Median 5×5 | 219 → 221 | +2 | 4 / 6 | 1 |
| Baseline Lab 2 | darken_4 → Gamma 2,0 (làm sáng) | 195 → 160 | -35 | 67 / 32 | 0.011 |
| Baseline Lab 2 | darken_4 → CLAHE clip 3 | 195 → 127 | -68 | 86 / 18 | 2e-10 |
| Baseline Lab 2 | low_contrast_4 → CLAHE clip 3 | 137 → 138 | +1 | 9 / 10 | 1 |
| Baseline Lab 2 | color_cast_4 → Gray-world | 75 → 127 | +52 | 56 / 108 | 0.0014 |
| MLP cải tiến | salt_pepper_3 → Median 3×3 | 304 → 322 | +18 | 6 / 24 | 0.026 |
| MLP cải tiến | salt_pepper_3 → Median 5×5 | 304 → 320 | +16 | 8 / 24 | 0.12 |
| MLP cải tiến | gaussian_noise_4 → Gaussian σ=1 | 290 → 290 | +0 | 4 / 4 | 1 |
| MLP cải tiến | gaussian_noise_4 → Bilateral d=9 | 290 → 293 | +3 | 4 / 7 | 1 |
| MLP cải tiến | gaussian_noise_4 → Median 5×5 | 290 → 292 | +2 | 11 / 13 | 1 |
| MLP cải tiến | darken_4 → Gamma 2,0 (làm sáng) | 92 → 121 | +29 | 15 / 44 | 0.0043 |
| MLP cải tiến | darken_4 → CLAHE clip 3 | 92 → 97 | +5 | 17 / 22 | 1 |
| MLP cải tiến | low_contrast_4 → CLAHE clip 3 | 148 → 156 | +8 | 32 / 40 | 1 |
| MLP cải tiến | color_cast_4 → Gray-world | 74 → 206 | +132 | 38 / 170 | 9.9e-20 |
| MobileNetV3 | salt_pepper_3 → Median 3×3 | 230 → 356 | +126 | 5 / 131 | 2.3e-31 |
| MobileNetV3 | salt_pepper_3 → Median 5×5 | 230 → 349 | +119 | 7 / 126 | 6.3e-28 |
| MobileNetV3 | gaussian_noise_4 → Gaussian σ=1 | 303 → 311 | +8 | 17 / 25 | 1 |
| MobileNetV3 | gaussian_noise_4 → Bilateral d=9 | 303 → 316 | +13 | 25 / 38 | 1 |
| MobileNetV3 | gaussian_noise_4 → Median 5×5 | 303 → 306 | +3 | 25 / 28 | 1 |
| MobileNetV3 | darken_4 → Gamma 2,0 (làm sáng) | 307 → 332 | +25 | 10 / 35 | 0.0049 |
| MobileNetV3 | darken_4 → CLAHE clip 3 | 307 → 295 | -12 | 35 / 23 | 1 |
| MobileNetV3 | low_contrast_4 → CLAHE clip 3 | 298 → 311 | +13 | 19 / 32 | 1 |
| MobileNetV3 | color_cast_4 → Gray-world | 321 → 346 | +25 | 8 / 33 | 0.0025 |
