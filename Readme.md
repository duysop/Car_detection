## Setup môi trường bằng anaconda
Sử dụng anaconda prompt
```
conda create -n cv python=3.9.5
conda activate cv
```
Cài thư viện
```
pip install -r requirements.txt
```
## Steps by steps

Raw data file
[raw.zip](https://drive.google.com/file/d/19zptcf0CuAJ183c_9Fu0dlWQs6PpxeVO/view?usp=sharing)

Để chạy thử trên các tham số khác nhau thì mỗi lần chạy xong cần phải xóa folder `non_cars` và chạy lại từ đầu vs các tham số thay đổi ở file `feature.py`


- Tạo dữ liệu cho Car
```
python car_img.py 
```
Parameters:
| Param | Description | Default | 
| ----- | ----------- | ------- |
| window_x | Kích thước x của window | 104 |
| window_y | Kích thước y của widow  | 56  |

- Tạo dữ liệu cho Non Car
```
python noncar_img.py --examples 16000
```
Bằng cách trượt windows trong ảnh không có ô tô để tạo ra negative samples. Tạo 16000 ảnh.
Parameters:
| Param | Description | Default | 
| ----- | ----------- | ------- |
| window_x | Kích thước x của window | 104 |
| window_y | Kích thước y của widow  | 56  |
| examples | Số lượng samples tạo ra  | 1e10  |
| downscale | Tỉ số giảm size ảnh mỗi lần (Pyramid Scale)  | 1.25  |
- Lấy chuyển ảnh thành features

Các tham số có thể thay đổi để đánh giá là --window_x, --window_y, --orientations, --signed_gradient, --gamma_correction

--window_x, --window_y phải là bội số của 8

Ví dụ như 

```
python feature.py --orientations 18 --signed_gradient True
```

Ở bước này chạy những tham số nào thì ở bước tạo `feature` lần 2 ở dưới cũng phải chạy những tham số đấy

Sau khi chạy bước này sẽ tạo ra file `data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmFalse`. or-18 là `orientations`, sgTrue là `signed_gradient`.

Parameters:
| Param | Description | Default | 
| ----- | ----------- | ------- |
| window_x | Kích thước x của window | 104 |
| window_y | Kích thước y của widow  | 56  |
| orientations | Số bins trong thuật toán HOG  | 9  |
| pixels_per_cell | Số pixels của 1 cell. 8 -> (8x8) cell | 8  |
| cells_per_block | Số cells trong 1 block. 2 -> (2x2) block  | 2  |
| signed_gradient | Sử dụng góc 0-180 nếu là False, 0-360 nếu là True | False |
| gamma_correction | Tính square root của ảnh trước khi tính HOG | False |
| hnm | Flag để xác định là đang lấy feature trước hay sau khi HNM | False |
- Train mô hình 

Ở bước trước ta có file `data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmFalse`. Do đó ta sẽ chạy train như sau

```
python svm_train.py --session "data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmFalse"
```

Parameters:
| Param | Description | Default | 
| ----- | ----------- | ------- |
| session | Tên file data đã dùng để train mô hình (Dùng để load mô hình dựa trên path) | data_w(104,56)_or9_ppc(8, 8)_cpb(2, 2)_sgFalse_gmFalse_hnmFalse |

Mất khoảng~ 2-5 p
- Tạo Hard Negative Samples 

Ở bước trước ta có file `data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmFalse` do đó chạy file HNM như sau.

```
python -W ignore hnm.py --session "data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmFalse"
```

File này sẽ thêm các hard negative samples để cải thiện mô hình

Output là file mới có đuôi là hmmTrue là `data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmTrue`


Parameters:
| Param | Description | Default | 
| ----- | ----------- | ------- |
| downscale | Tỉ số giảm size ảnh mỗi lần (Pyramid Scale)  | 1.25  |
| session | Tên file data đã dùng để train mô hình (Dùng để load mô hình dựa trên path) | data_w(104,56)_or9_ppc(8, 8)_cpb(2, 2)_sgFalse_gmFalse_hnmFalse  |
| examples | Số lượng samples tạo ra  | 1e10  |
- Lấy feature của tất cả ảnh sau bước HNM 

Ở trên mình dùng 2 tham số là orientations 18 và signed_gradient do đó chạy

```
python feature.py --hnm True --orientations 18 --signed_gradient True
```
- Train mô hình lần 2
```
python svm_train.py --session "data_w(104,56)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmTrue"
```
- Detector (Chỉ để test thử coi detect có được hay k nên k cần chạy nhiều)
```
python detector.py --image "raw/test_case/test_case0.jpg"
```
Có thể chỉnh tham số threshold ở trong file nếu có nhiều cửa sổ bị sai

Parameters:
| Param | Description | Default | 
| ----- | ----------- | ------- |
| image | Đường dẫn ảnh cần detect  | raw/test_case/test_case0.jpg  |
| session | Tên file data đã dùng để train mô hình (Dùng để load mô hình dựa trên path) | data_w(104,56)_or9_ppc(8, 8)_cpb(2, 2)_sgFalse_gmFalse_hnmFalse  |

- Get Groud_Truth 
Vào notebook groundtruths.ipynb chạy để tạo ground_truths ở data/ground_truths

Bước này chỉ cần chạy 1 lần duy nhất nên nếu sửa đổi tham số thì k cần chạy lần 2 nữa.

- Get Detection Bounding Boxes

Vào trong file `detection.py` sửa biến `session` thành trên file data vừa được tạo ra ở bước `Lấy feature của tất cả ảnh sau bước HNM`

Đuôi của file phải là `hnmTrue`

Xong rồi chạy
`python detections.py`
Đợi 20
Output của file này là 1 detection bounding boxes dạng .txt ở data/evaluation/{sesssion}

- Evaluate

Vào trong file `detection.py` 
sửa biến session thành trên file data vừa được tạo ra ở bước `Lấy feature của tất cả ảnh sau bước HNM`

Đuôi của file phải là `hnmTrue`
```
python evaluate.py
```

Output của file này là 1 file txt là 1 file pkl ở data/evaluation/{sesssion}

Xem file .txt để biết AP: Average Precision
