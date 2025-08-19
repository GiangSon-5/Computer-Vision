# Convolution (Cross-correlation) và Các Kernel Phổ Biến trong Computer Vision

Trong YOLO và hầu hết các CNN, "convolution" thực chất là **cross-correlation**:

$$
S(x, y) = \sum_m \sum_n I(x+m, y+n) \cdot K(m, n)
$$

- $I$: ảnh đầu vào  
- $K$: kernel (mặt nạ lọc, thường có kích thước 3x3 hoặc 5x5)  
- $S$: ảnh đầu ra  

---

## 1. Một số kernel phổ biến

### a) Laplacian (lọc biên toàn cục)

$$
K = \begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0
\end{bmatrix}
$$

---

### b) Horizontal Filter (biên ngang)

$$
K = \begin{bmatrix}
-1 & -1 & -1 \\
2 & 2 & 2 \\
-1 & -1 & -1
\end{bmatrix}
$$

---

### c) Blur (làm mờ trung bình)

$$
K = \frac{1}{9} \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

---

## 2. Ví dụ minh họa: Ma trận 5x5 lọc với kernel 3x3

Giả sử ảnh đầu vào:

$$
I = \begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\
5 & 6 & 7 & 8 & 9 \\
9 & 8 & 7 & 6 & 5 \\
4 & 3 & 2 & 1 & 0 \\
0 & 1 & 2 & 3 & 4
\end{bmatrix}
$$

Chọn vị trí **pixel trung tâm $(x=2, y=2)$**, tức hàng 3, cột 3 (theo chỉ số 1-based).  
Vùng lân cận 3x3 quanh điểm này là:

$$
I_{local} = \begin{bmatrix}
6 & 7 & 8 \\
8 & 7 & 6 \\
3 & 2 & 1
\end{bmatrix}
$$

---

![Horizontal](..\..\imgs\Horizontal.jpg)

### a) Horizontal Filter (biên ngang)

Kernel:

$$
K = \begin{bmatrix}
-1 & -1 & -1 \\
2 & 2 & 2 \\
-1 & -1 & -1
\end{bmatrix}
$$

Tính:

$$
S_{edge}(2,2) = (6)(-1)+(7)(-1)+(8)(-1) \\
+ (8)(2)+(7)(2)+(6)(2) \\
+ (3)(-1)+(2)(-1)+(1)(-1)
$$

Kết quả:

$$
S_{edge}(2,2) = 15
$$

> **Ý nghĩa**: Giá trị dương lớn cho thấy có đường biên ngang mạnh tại vùng này.

---

### b) Laplacian (lọc biên toàn cục)

Kernel:

$$
K = \begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0
\end{bmatrix}
$$

Tính:

$$
S_{lap}(2,2) = (7)(-1) + (8)(-1) + (8)(-1) + (6)(-1) + (7)(4)
$$

Kết quả:

$$
S_{lap}(2,2) = -7 -8 -8 -6 + 28 = -1
$$

> **Ý nghĩa**: Kết quả gần 0 → tại vị trí này không có biên rõ rệt theo Laplacian.

---

### c) Blur (làm mờ trung bình)

Kernel:

$$
K = \frac{1}{9} \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

Tính trung bình 9 giá trị trong $I_{local}$:

$$
S_{blur}(2,2) = \frac{6+7+8+8+7+6+3+2+1}{9}
$$

Kết quả:

$$
S_{blur}(2,2) = \frac{48}{9} \approx 5.33
$$

> **Ý nghĩa**: Làm mờ → giá trị pixel trở thành trung bình, giúp giảm nhiễu.

---

## 3. Nhận xét

- **Edge Filter (custom)**: nhấn mạnh cạnh ngang, kết quả lớn (15).  
- **Laplacian**: bắt biên đa hướng, tại điểm này gần như không có biên rõ ($-1$).  
- **Blur**: làm mờ, pixel trung tâm thành giá trị trung bình ($5.33$).

👉 Ba bộ lọc cùng áp dụng trên một vùng, nhưng kết quả khác nhau hoàn toàn → cho thấy mỗi kernel "nhìn ảnh" theo một cách riêng để trích xuất đặc trưng.

---

## 3. Kết luận

- Convolution (cross-correlation) là phép nhân trượt kernel trên ảnh.  
- Các kernel phổ biến:  
  - **Laplacian** → nhấn mạnh biên toàn cục  
  - **Edge Filter (custom)** → phát hiện biên ngang rõ rệt  
  - **Blur** → làm mờ, giảm nhiễu  
- Ví dụ với ma trận 5x5 cho thấy cách tính cụ thể từng giá trị pixel sau khi lọc.

---

## 4. Pipeline: Ảnh gốc → Blur → Edge Filter

Trong thực tế, để **phát hiện biên** tốt hơn, ta không áp dụng trực tiếp Sobel hay Laplacian lên ảnh gốc, mà thường thêm bước **làm mờ (Blur)** trước.  

### Lý do:
- Ảnh gốc thường có **nhiễu (noise)**: pixel đơn lẻ sáng/tối bất thường.  
- Nếu áp ngay edge filter → nhiễu này cũng bị coi là "biên", tạo ra biên giả.  
- Blur kernel giúp **làm trơn (smooth)** cục bộ, giảm nhiễu, giữ lại cấu trúc lớn → biên thật được nhấn mạnh hơn.

---

### Pipeline cơ bản

$$
I_{edge} = (I * K_{blur}) * K_{edge}
$$

Trong đó:
- $I$: ảnh gốc  
- $K_{blur}$: kernel làm mờ (ví dụ trung bình 3x3)  
- $K_{edge}$: kernel biên (Sobel, Laplacian, hoặc custom)  

---

### Ví dụ minh họa

1. **Ảnh gốc**: có nhiều chi tiết và nhiễu.  
2. **Blur (3x3 mean filter)**: giảm nhiễu, làm mượt ảnh.  
3. **Edge Filter (ví dụ Laplacian)**: phát hiện biên rõ ràng, ít bị rối bởi nhiễu.

Kết quả:  
- Nếu bỏ bước Blur → biên xuất hiện cả ở vùng nhiễu (biên giả).  
- Nếu có Blur → biên chủ yếu ở vùng thay đổi thật sự (vật thể, contour).

---

### Minh họa toán học

## 4. Blur để giảm nhiễu trước khi Edge Detection  

### 1. Nhiễu ảnh là gì?  
Trong ảnh thật thường tồn tại **pixel nhiễu**:  
- Điểm sáng bất thường  
- Điểm tối bất thường  
- Dao động ngẫu nhiên về cường độ  

Ví dụ: trong ảnh xám 8-bit (0–255), vùng xung quanh có giá trị ~100–120, nhưng xuất hiện một pixel = 250 → đó là **nhiễu**.

---

### 2. Blur hoạt động như thế nào?  
Kernel làm mờ (ví dụ **mean filter 3x3**) lấy giá trị trung bình của các điểm lân cận:  

$$
I'(x,y) = \frac{1}{N} \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x+i, y+j)
$$  

Trong đó $N$ là số phần tử trong kernel (ví dụ 9 cho kernel 3x3).  

➡ Nếu có một pixel nhiễu (rất khác biệt so với hàng xóm), giá trị đó sẽ bị **pha loãng** trong phép trung bình → giảm tác động của nhiễu.  

---

### 3. Ví dụ minh họa  

Ảnh gốc 3x3 có một pixel nhiễu:  

$$
I = \begin{bmatrix}
100 & 102 & 101 \\
99 & 250 & 98 \\
100 & 101 & 99
\end{bmatrix}
$$  

- Pixel trung tâm = 250, rõ ràng bất thường (các giá trị khác chỉ quanh ~100).  
- Nếu không lọc, điểm này sẽ hiện lên như một chấm trắng chói.  

Áp dụng **mean blur 3x3**:  

$$
I'(1,1) = \frac{100+102+101+99+250+98+100+101+99}{9} = \frac{1050}{9} \approx 117
$$  

👉 Giá trị 250 đã bị “pha loãng”, giảm về ~117, gần hơn với bối cảnh (~100).  

---

### 4. Ý nghĩa trong edge detection  
- **Không làm mờ trước**: nhiễu tạo ra các biên giả khi dùng Sobel, Laplacian → ảnh bị rối, nhiều cạnh không thật.  
- **Có làm mờ trước**: nhiễu giảm, biên thật (contour) giữ lại rõ hơn → giúp Edge Detection ổn định và chính xác hơn.  


---

## 5. Tổng kết

- Pipeline chuẩn trong xử lý ảnh cổ điển:  
  **Ảnh gốc → Blur (giảm nhiễu) → Edge Filter (Sobel/Laplacian/custom)**  
- Blur đóng vai trò như "bộ lọc trước" để biên phát hiện ra **ít nhiễu, chính xác hơn**.

