

### 1. Nguyên tắc tính output size của pooling/convolution

Công thức tổng quát:

$$
\text{out} = \left\lfloor \frac{in + 2p - k}{s} \right\rfloor + 1
$$

Trong đó:

* `in`: kích thước input (ở đây là 20).
* `k`: kernel size (ở đây là 5).
* `s`: stride (ở đây = 1).
* `p`: padding.

---

### 2. SPPF dùng padding “same”

Trong YOLOv5/YOLOv8, MaxPool trong SPPF được định nghĩa với:

* `kernel = 5`
* `stride = 1`
* `padding = 2`

Thay vào công thức:

$$
\text{out} = \left\lfloor \frac{20 + 2*2 - 5}{1} \right\rfloor + 1 = 20
$$

👉 Kết quả: giữ nguyên kích thước 20.
---

## 1. Trước hết, kích thước không gian (H, W)

* Ban đầu `[N, 512, 20, 20]`.
* Mỗi lần **MaxPool2D(k=5, stride=1, padding=2)** thì **H, W giữ nguyên = 20×20**.
* Vậy: *không gian (spatial size) không đổi*.

---

## 2. Cái gì tăng?

Chính là **số kênh (C)** tăng.

Cụ thể:

* `x1` có 512 kênh.
* `m1, m2, m3` cũng mỗi cái 512 kênh.
* Khi `torch.cat([x1, m1, m2, m3], dim=1)` thì số kênh **cộng dồn lại**:

$$
C_{out} = 512 + 512 + 512 + 512 = 512 \times 4 = 2048
$$

---

## 3. Tại sao nhân 4?

Vì bạn **ghép 4 tensor theo chiều channel**:

* 1 tensor gốc (x1)
* 3 tensor từ pooling (m1, m2, m3)

👉 Tổng cộng là 4 “bản thể” của cùng feature map, mỗi bản thể vẫn giữ nguyên spatial size (20×20), chỉ khác về receptive field.

---

## 4. Sau đó tại sao Conv1×1 đưa về 1024?

* Ghép lại thành `[N, 2048, 20, 20]` sẽ rất nặng (channel quá nhiều).
* Conv1×1 (`cv2`) được dùng để **giảm số kênh từ 2048 → 1024**, giữ lại thông tin quan trọng nhưng giảm chi phí tính toán.

---

✅ Tóm lại:

* **Tăng nhân 4** là do concat 4 feature maps theo channel.
* Công thức:

$$
C_{out} = C_{in} \times (1 + \text{số lần pooling})
$$

Trong case này:

$$
C_{out} = 512 \times (1+3) = 2048
$$

* Sau đó Conv1×1 giảm còn 1024.


---





---

## 2. Công thức tăng receptive field

Giả sử ta có:

* kernel size = $k$
* stride = $s$
* receptive field lớp trước = $RF\_{prev}$

Công thức tổng quát:

$$
RF_{new} = RF_{prev} + (k - 1) \times \text{jump}_{prev}
$$

Trong đó:

* $\text{jump}\_{prev}$ = khoảng cách 2 vị trí liên tiếp trong RF trước (ở stride=1 thì jump=1).
* Với maxpool k=5, s=1 → $(k-1)=4$, nên mỗi lần RF tăng thêm 4.

---

## 3. Áp dụng cho SPPF

* Ban đầu: $RF = 1$ (mỗi pixel nhìn đúng 1 pixel input).
* Sau 1 pool (k=5, s=1):

$$
RF = 1 + (5-1) \times 1 = 5
$$

* Sau 2 pool:

$$
RF = 5 + (5-1) \times 1 = 9
$$

* Sau 3 pool:

$$
RF = 9 + (5-1) \times 1 = 13
$$

👉 Kết quả: 5 → 9 → 13.



