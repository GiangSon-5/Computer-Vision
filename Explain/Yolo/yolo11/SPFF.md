# SPPF (ultralytics.nn.modules.block.SPPF) 

---

## 0) Code `__init__` (nhìn nhanh)
>  (Đoạn này là trích ý chính từ `ultralytics/nn/modules/block.py` — hiển thị dạng preformatted)
>     
>     class SPPF(nn.Module):
>         def __init__(self, c1: int, c2: int, k: int = 5):
>             super().__init__()
>             c_ = c1 // 2               # hidden channels
>             self.cv1 = Conv(c1, c_, 1, 1)
>             self.cv2 = Conv(c_ * 4, c2, 1, 1)
>             self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

> *Ghi chú:* `cv1` giảm kênh, `m` là MaxPool2d (stride=1), `cv2` gộp 4×kênh về `c2`. Trong `forward`, `m` thường được gọi 3 lần: `y1 = m(x); y2 = m(y1); y3 = m(y2)` rồi `cat([x,y1,y2,y3], dim=1)`.

---

## 1) Mục đích ngắn gọn
- *SPPF* tăng **receptive field** (bối cảnh) theo nhiều tỉ lệ mà **không giảm** độ phân giải không gian (H×W).  
- Cho phép Head/Neck sử dụng cả thông tin *local* và *global* tại mỗi pixel.  
- *Có thể dùng độc lập* — C1/C2 (CSP blocks) là các khối xử lý kênh bổ sung nhưng **không bắt buộc** để SPPF chạy.

---

## 2) Công thức tổng quan 

$$
c' = \left\lfloor \frac{c_1}{2} \right\rfloor
$$

$$
X' = \text{cv1}(X) \quad \text{shape } (N, c', H, W)
$$

$$
P_1 = \text{MaxPool}(X', k),\quad P_2 = \text{MaxPool}(P_1, k),\quad P_3 = \text{MaxPool}(P_2, k)
$$

$$
\text{Concat} = \text{ConcatChannels}(X', P_1, P_2, P_3) \quad \text{shape } (N, 4c', H, W)
$$

$$
Y = \text{cv2}(\text{Concat}) \quad \text{shape } (N, c_2, H, W)
$$

Với `Conv1x1` tại mỗi vị trí `(n,o,h,w)`:

$$
Y_{n,o,h,w} = \sum_{i=0}^{C_{\text{in}}-1} W_{o,i}\;X_{n,i,h,w} + b_o
$$

---

## 3) Pseudocode forward (ý nghĩa)
>     
>     def forward(X):
>         x = cv1(X)            # reduce channels -> X'
>         y1 = m(x)             # MaxPool
>         y2 = m(y1)            # MaxPool
>         y3 = m(y2)            # MaxPool
>         out = cv2(concat(x, y1, y2, y3))
>         return out

---
# Ví dụ SPPF — *Chỉ ví dụ (chi tiết từng phép toán, giải thích cv2 và "gạch đầu dòng")*

> Yêu cầu: Bỏ lý thuyết, chỉ làm ví dụ 2D chi tiết.  
> Mọi bước tính hiện rõ, giải thích `cv2` xuất phát từ đâu, và giải thích "gạch đầu dòng" (bullet points).

---

## Thiết lập ví dụ 
- Batch N = 1 (bỏ chỉ số n).  
- Input channels `c1 = 2`, output channels `c2 = 1`.  
- H = W = 4.  
- Kernel pooling `k = 3`, stride=1, padding=1 (padding giá trị = 0).  
- `cv1` (Conv1×1 → 1 channel) trọng số: `w_A = 1.0`, `w_B = 0.5`, bias = 0.  
- `cv2` (Conv1×1 gộp 4 channel → 1) trọng số theo channel: `[α1, α2, α3, α4] = [1.0, 0.5, 0.1, 0.1]`, bias = 0.  
  - **Ghi chú về `cv2`:** trong mô hình thực tế, `α1..α4` là các trọng số học được (learnable). Ở ví dụ này ta **giả sử** giá trị để minh họa cách `cv2` kết hợp bốn nguồn (Y, P1, P2, P3).

---

## 4.1 Input (2 channel)

Channel A:
```lua
A =
[1 2 3 0
4 5 6 1
7 8 9 2
0 1 2 3]
```


Channel B:

```lua
B =
[0 1 0 1
2 1 2 1
0 1 0 1
1 0 1 0]
```


---

## 4.2 Bước 1 — `cv1` (Conv1×1 giảm 2ch → 1ch)

- Công thức tại pixel (i,j):

$$
Y(i,j) = 1.0\cdot A(i,j) + 0.5\cdot B(i,j)
$$

- Tính 0.5 * B và Y (đã có ở ví dụ trước):

```lua
0.5 * B =
[0 0.5 0 0.5
1 0.5 1 0.5
0 0.5 0 0.5
0.5 0 0.5 0 ]

Y = A + 0.5*B =
[1 2.5 3 0.5
5 5.5 7 1.5
7 8.5 9 2.5
0.5 1 2.5 3 ]
```


---
## 4.3 Bước 2 — P1 = MaxPool(Y, k=3, s=1, p=1)

- **Công thức:**

$$
P1(i,j) = \max_{(u,v) \in \Omega_{3\times3}(i,j)} Y(u,v)
$$

Trong đó $\Omega_{3\times3}(i,j)$ là cửa sổ 3×3 bao quanh vị trí $(i,j)$, với **padding ngoài biên = 0**.

---

### Ví dụ tính chi tiết

**P1(0,0):**  
Cửa sổ 3×3 (pad 0 ở trên và trái):

|     |     |     |
|-----|-----|-----|
| 0   | 0   | 0   |
| 0   | 1.0 | 2.5 |
| 0   | 5.0 | 5.5 |

→ Max = **5.5**

---

**P1(0,1):**  
Cửa sổ 3×3 (hàng đầu pad 0):

|     |     |     |
|-----|-----|-----|
| 0   | 0   | 0   |
| 1.0 | 2.5 | 3.0 |
| 5.0 | 5.5 | 7.0 |

→ Max = **7.0**

---

**P1(1,1):**  
Cửa sổ 3×3 đầy đủ trong ma trận Y:

|     |     |     |
|-----|-----|-----|
| 1.0 | 2.5 | 3.0 |
| 5.0 | 5.5 | 7.0 |
| 7.0 | 8.5 | 9.0 |

→ Max = **9.0**

---

### Kết quả toàn bộ ma trận P1

| 5.5 | 7.0 | 7.0 | 7.0 |
|-----|-----|-----|-----|
| 8.5 | 9.0 | 9.0 | 9.0 |
| 8.5 | 9.0 | 9.0 | 9.0 |
| 8.5 | 9.0 | 9.0 | 9.0 |

---

## 4.4 Bước 3 — P2 = MaxPool(P1), P3 = MaxPool(P2)

- Ta tiếp tục pooling với cùng tham số (k=3, s=1, p=1).  
- Vì trong P1 đã có nhiều giá trị **9.0**, nên sau pooling lần 2 toàn bộ ma trận trở thành **9.0**.  
- Pooling lần 3 giữ nguyên.

### Kết quả

| 9.0 | 9.0 | 9.0 | 9.0 |
|-----|-----|-----|-----|
| 9.0 | 9.0 | 9.0 | 9.0 |
| 9.0 | 9.0 | 9.0 | 9.0 |
| 9.0 | 9.0 | 9.0 | 9.0 |


---
## 4.5 Bước 4 — Concat theo channel

- Sau khi có kết quả từ các bước trước, ta gom 4 ma trận thành một tensor nhiều kênh (multi-channel).
- Cụ thể:

  - **Channel 1** = Y  
  - **Channel 2** = P1  
  - **Channel 3** = P2  
  - **Channel 4** = P3  

> Như vậy tại mỗi vị trí `(i,j)` ta có một vector 4 chiều:  
> `[Y(i,j), P1(i,j), P2(i,j), P3(i,j)]`

---

## 4.6 Bước 5 — `cv2` (Conv1×1 gộp 4 channel → 1 channel)

### Công thức tổng quát

Tại mỗi pixel `(i,j)`, phép tích chập 1×1 sẽ tính:

$$
Z(i,j) = \alpha_1 \cdot Y(i,j) \;+\; \alpha_2 \cdot P1(i,j) \;+\; \alpha_3 \cdot P2(i,j) \;+\; \alpha_4 \cdot P3(i,j)
$$

- Trong đó:
  - `Y, P1, P2, P3` là giá trị của từng channel tại cùng vị trí `(i,j)`.
  - `α1, α2, α3, α4` là trọng số của kernel Conv1×1 (tham số học được trong quá trình training).
  - Ở đây **bias = 0** để đơn giản hóa.

---

### Trọng số giả định (ví dụ)

Ta giả sử vector trọng số:

```lua
α = [α1, α2, α3, α4] = [1.0, 0.5, 0.1, 0.1]
```


---

### Lý do xuất hiện hằng số S = 1.8

- Ở bước pooling trước ta có:

```lua
P2 = P3 = ma trận toàn giá trị 9
```


- Do đó, với mọi vị trí `(i,j)`:

```lua
α3 * P2(i,j) + α4 * P3(i,j)
= 0.1 * 9 + 0.1 * 9
= 1.8
```


- Thành phần này không phụ thuộc vào `(i,j)` nên có thể gộp lại thành một **hằng số bổ sung**:


```lua
S = 1.8
```


---

### Công thức rút gọn

Khi thay S vào, công thức trở thành:

$$
Z(i,j) = \alpha_1 \cdot Y(i,j) \;+\; \alpha_2 \cdot P1(i,j) \;+\; S
$$

Với:

- `α1 = 1.0`  
- `α2 = 0.5`  
- `S = 1.8`

---

### Ví dụ tính tại một pixel

- Tại vị trí `(0,0)`:
  - `Y(0,0) = 1.0`
  - `P1(0,0) = 5.5`
  - Áp dụng công thức:

    ```
    Z(0,0) = 1.0 * 1.0 + 0.5 * 5.5 + 1.8
           = 1.0 + 2.75 + 1.8
           = 5.55
    ```

---


---

## Tính **từng ô** (từng phép cộng, từng chữ số) — **toàn bộ ma trận Z**

> Ta sẽ liệt kê Z(i,j) cho mọi (i,j) theo quy ước hàng rồi cột (0-indexed trong tính nhưng hiển thị ma trận).

**Ô (0,0):**
- Y(0,0) = 1  
- 0.5*P1(0,0) = 0.5 * 5.5 = 2.75  
- + S = 1.8  
- Z(0,0) = 1 + 2.75 + 1.8 = **5.55**

**Ô (0,1):**
- Y(0,1) = 2.5  
- 0.5*P1(0,1) = 0.5 * 7 = 3.5  
- + S = 1.8  
- Z(0,1) = 2.5 + 3.5 + 1.8 = **7.80**

**Ô (0,2):**
- Y(0,2) = 3  
- 0.5*P1(0,2) = 0.5 * 7 = 3.5  
- + S = 1.8  
- Z(0,2) = 3 + 3.5 + 1.8 = **8.30**

**Ô (0,3):**
- Y(0,3) = 0.5  
- 0.5*P1(0,3) = 0.5 * 7 = 3.5  
- + S = 1.8  
- Z(0,3) = 0.5 + 3.5 + 1.8 = **5.80**

---

**Ô (1,0):**
- Y(1,0) = 5  
- 0.5*P1(1,0) = 0.5 * 8.5 = 4.25  
- + S = 1.8  
- Z(1,0) = 5 + 4.25 + 1.8 = **11.05**

**Ô (1,1):**
- Y(1,1) = 5.5  
- 0.5*P1(1,1) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(1,1) = 5.5 + 4.5 + 1.8 = **11.80**

**Ô (1,2):**
- Y(1,2) = 7  
- 0.5*P1(1,2) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(1,2) = 7 + 4.5 + 1.8 = **13.30**

**Ô (1,3):**
- Y(1,3) = 1.5  
- 0.5*P1(1,3) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(1,3) = 1.5 + 4.5 + 1.8 = **7.80**

---

**Ô (2,0):**
- Y(2,0) = 7  
- 0.5*P1(2,0) = 0.5 * 8.5 = 4.25  
- + S = 1.8  
- Z(2,0) = 7 + 4.25 + 1.8 = **13.05**

**Ô (2,1):**
- Y(2,1) = 8.5  
- 0.5*P1(2,1) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(2,1) = 8.5 + 4.5 + 1.8 = **14.80**

**Ô (2,2):**
- Y(2,2) = 9  
- 0.5*P1(2,2) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(2,2) = 9 + 4.5 + 1.8 = **15.30**

**Ô (2,3):**
- Y(2,3) = 2.5  
- 0.5*P1(2,3) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(2,3) = 2.5 + 4.5 + 1.8 = **8.80**

---

**Ô (3,0):**
- Y(3,0) = 0.5  
- 0.5*P1(3,0) = 0.5 * 8.5 = 4.25  
- + S = 1.8  
- Z(3,0) = 0.5 + 4.25 + 1.8 = **6.55**

**Ô (3,1):**
- Y(3,1) = 1  
- 0.5*P1(3,1) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(3,1) = 1 + 4.5 + 1.8 = **7.30**

**Ô (3,2):**
- Y(3,2) = 2.5  
- 0.5*P1(3,2) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(3,2) = 2.5 + 4.5 + 1.8 = **8.80**

**Ô (3,3):**
- Y(3,3) = 3  
- 0.5*P1(3,3) = 0.5 * 9 = 4.5  
- + S = 1.8  
- Z(3,3) = 3 + 4.5 + 1.8 = **9.30**

---

## Ma trận kết quả Z (làm tròn 2 chữ số)

```lua
Z =
[ 5.55 7.80 8.30 5.80
11.05 11.80 13.30 7.80
13.05 14.80 15.30 8.80
6.55 7.30 8.80 9.30 ]
```