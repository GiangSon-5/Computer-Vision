# Attention Module trong YOLO — Giải thích chi tiết + Ví dụ minh họa

## 1. Giới thiệu

Mục tiêu của **Attention** là cho phép mỗi vị trí (pixel/patch) trong ảnh **liên kết** với tất cả các vị trí khác, chứ không chỉ nhìn local như convolution.  
Nhờ đó, mô hình học được cả:

- Quan hệ gần (local features).
- Quan hệ xa (global context).

---

## 2. Tham số quan trọng

- `dim`: tổng số kênh của feature đầu vào.
- `num_heads`: chia kênh thành nhiều *head* để học song song.
- `attn_ratio`: định nghĩa kích thước của Query/Key so với Value.

👉 Hiểu nôm na: **Value** chứa thông tin "nội dung", còn **Query/Key** chỉ đóng vai trò "so khớp mức độ liên quan".  
Vì thế thường cho Q,K nhỏ hơn V để giảm chi phí tính toán.

---

## 3. Các thuộc tính chính

- **`head_dim = dim // num_heads`**  
  → mỗi head xử lý một lát cắt kênh riêng biệt.

- **`key_dim = int(head_dim * attn_ratio)`**  
  → quyết định kích thước Q, K.

- **`scale = 1 / \sqrt{key_dim}`**  
  → nếu không có scale, khi tính $QK^T$, giá trị có thể lớn → softmax saturate → gradient vanish.

- **`qkv`**: Conv1×1 tạo Q, K, V cùng lúc.

- **`proj`**: Conv1×1 hợp nhất kết quả từ nhiều head.

- **`pe`**: Conv3×3 (group conv) để bổ sung thông tin vị trí (positional encoding).

---

## 4. Luồng xử lý (forward)

### Bước 1 — Sinh Q, K, V

- Input: $x$ có shape $(B, C, H, W)$
- Sau `Conv1×1`: tạo tensor gộp `[Q, K, V]`.
- Split:

$$
\begin{aligned}
Q &: (B, \text{num}_{\text{heads}}, \text{key}_{\text{dim}}, N) \\
K &: (B, \text{num}_{\text{heads}}, \text{key}_{\text{dim}}, N) \\
V &: (B, \text{num}_{\text{heads}}, \text{head}_{\text{dim}}, N)
\end{aligned}
$$



với $N = H \times W$ (tổng số pixel).

---

### Bước 2 — Attention Score

Tính độ tương tự (similarity) giữa Q và K:

$$
S = Q^T K \cdot \text{scale}
$$

- $S$ có shape $(B, \text{num\_heads}, N, N)$.
- Mỗi phần tử $s_{ij}$ = mức độ liên quan của vị trí $i$ đến vị trí $j$.

Chuẩn hóa softmax theo từng hàng:

$$
\text{attn}_{ij} = \frac{\exp(s_{ij})}{\sum_{j} \exp(s_{ij})}
$$

---

### Bước 3 — Kết hợp với V

Tạo output bằng cách nhân attention với V:

$$
O = \text{attn} \cdot V^T
$$

---

### Bước 4 — Positional Encoding (PE)

Self-attention **không biết vị trí tuyệt đối** → cần thêm PE:

$$
O' = O + \text{PE}(O)
$$

---

### Bước 5 — Projection

Hợp nhất các head và đưa về số kênh ban đầu:

$$
\text{Out} = \text{Conv}_{1\times1}(O')
$$

---

## 5. Ví dụ minh họa số học

Giả sử **feature map** nhỏ: **3x3** ($H=3, W=3, N=9$), với **4 kênh** ($\text{dim}=4$).  
Dùng **2 head** (`num_heads=2`), `attn_ratio=0.5` → `head_dim = 4 // 2 = 2`, `key_dim = int(2 * 0.5) = 1`, `scale = 1 / \sqrt{1} = 1`.

**Input X** (B=1, C=4, H=3, W=3): Các kênh tăng dần để dễ theo dõi.

$$
X[:,0,:,:] = 
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix},\quad
X[:,1,:,:] = 
\begin{bmatrix}
10 & 11 & 12 \\
13 & 14 & 15 \\
16 & 17 & 18
\end{bmatrix},\quad
X[:,2,:,:] = 
\begin{bmatrix}
19 & 20 & 21 \\
22 & 23 & 24 \\
25 & 26 & 27
\end{bmatrix},\quad
X[:,3,:,:] = 
\begin{bmatrix}
28 & 29 & 30 \\
31 & 32 & 33 \\
34 & 35 & 36
\end{bmatrix}
$$

Để minh họa, chúng ta sử dụng trọng số được khởi tạo ngẫu nhiên (với seed 42 để tái tạo), sử dụng phân phối normal (mean=0, std=0.02) cho các conv layers.

---

### 5.1 Tạo Q, K, V 

⚡ **[VÍ DỤ Q, K, V được chọn như nào ](../yolo11/EX/attention_module_yolo_3x3_example.md)**

Sau Conv1×1 (qkv), tensor gộp QKV có shape (1,8,3,3). Các giá trị tính được như sau (làm tròn đến 4 chữ số):

$$
\text{QKV}[:,0,:,:] = 
\begin{bmatrix}
-0.5008 & -0.4566 & -0.4124 \\
-0.3683 & -0.3241 & -0.2799 \\
-0.2357 & -0.1915 & -0.1473
\end{bmatrix},\quad
\text{QKV}[:,1,:,:] = 
\begin{bmatrix}
-1.1483 & -1.1924 & -1.2365 \\
-1.2806 & -1.3246 & -1.3687 \\
-1.4128 & -1.4569 & -1.5009
\end{bmatrix},
$$

(tương tự cho các kênh còn lại: -0.6205 đến -0.7644 cho kênh 2, 0.0084 đến -0.1986 cho kênh 3, 0.0581 đến 0.2861 cho kênh 4, 1.4460 đến 1.8942 cho kênh 5, 1.2643 đến 1.9876 cho kênh 6, 0.3896 đến 0.4566 cho kênh 7).


Sau reshape và split:

- **Q** (shape [1,2,1,9]): Flatten theo vị trí.

  - Head 0: [-0.5008, -0.4566, -0.4124, -0.3683, -0.3241, -0.2799, -0.2357, -0.1915, -0.1473]

  - Head 1: [0.0581, 0.0866, 0.1151, 0.1436, 0.1721, 0.2006, 0.2291, 0.2576, 0.2861]

- **K** (shape [1,2,1,9]):

  - Head 0: [-1.1483, -1.1924, -1.2365, -1.2806, -1.3246, -1.3687, -1.4128, -1.4569, -1.5009]

  - Head 1: [1.4460, 1.5020, 1.5580, 1.6140, 1.6701, 1.7261, 1.7821, 1.8382, 1.8942]

- **V** (shape [1,2,2,9]):

  Head 0: 

  - Chiều 0: [-0.6205, -0.6384, -0.6564, -0.6744, -0.6924, -0.7104, -0.7284, -0.7464, -0.7644]

  - Chiều 1: [0.0084, -0.0175, -0.0434, -0.0693, -0.0951, -0.1210, -0.1469, -0.1728, -0.1986]

  Head 1: 

  - Chiều 0: [1.2643, 1.3547, 1.4451, 1.5355, 1.6260, 1.7164, 1.8068, 1.8972, 1.9876]

  - Chiều 1: [0.3896, 0.3980, 0.4064, 0.4147, 0.4231, 0.4315, 0.4399, 0.4482, 0.4566]

---

### 5.2 Tính similarity $S = Q^T K \cdot \text{scale}$

Vì key_dim=1, $Q^T K$ là ma trận [9,9] cho mỗi head, với $S_{ij} = q_i \cdot k_j \cdot 1$.

Sau đó áp dụng softmax để được attn (làm tròn đến 4 chữ số). Dưới đây là attn cho Head 0 (ma trận 9x9):

$$
A_0 \approx
\begin{bmatrix}
0.1016 & 0.1038 & 0.1061 & 0.1085 & 0.1109 & 0.1134 & 0.1159 & 0.1185 & 0.1212 \\
0.1024 & 0.1045 & 0.1066 & 0.1088 & 0.1110 & 0.1132 & 0.1155 & 0.1179 & 0.1203 \\
0.1032 & 0.1051 & 0.1070 & 0.1090 & 0.1110 & 0.1130 & 0.1151 & 0.1172 & 0.1194 \\
0.1040 & 0.1057 & 0.1075 & 0.1092 & 0.1110 & 0.1128 & 0.1147 & 0.1166 & 0.1185 \\
0.1049 & 0.1064 & 0.1079 & 0.1095 & 0.1110 & 0.1126 & 0.1143 & 0.1159 & 0.1176 \\
0.1057 & 0.1070 & 0.1083 & 0.1097 & 0.1111 & 0.1124 & 0.1138 & 0.1152 & 0.1167 \\
0.1066 & 0.1077 & 0.1088 & 0.1099 & 0.1111 & 0.1122 & 0.1134 & 0.1146 & 0.1158 \\
0.1074 & 0.1083 & 0.1092 & 0.1102 & 0.1111 & 0.1120 & 0.1130 & 0.1139 & 0.1149 \\
0.1082 & 0.1090 & 0.1097 & 0.1104 & 0.1111 & 0.1118 & 0.1125 & 0.1133 & 0.1140
\end{bmatrix}
$$

Attn cho Head 1 (tương tự, nhưng giá trị khác):

$$
A_1 \approx
\begin{bmatrix}
0.1097 & 0.1100 & 0.1104 & 0.1107 & 0.1111 & 0.1115 & 0.1118 & 0.1122 & 0.1126 \\
0.1090 & 0.1095 & 0.1100 & 0.1106 & 0.1111 & 0.1116 & 0.1122 & 0.1127 & 0.1133 \\
0.1083 & 0.1090 & 0.1097 & 0.1104 & 0.1111 & 0.1118 & 0.1125 & 0.1133 & 0.1140 \\
0.1076 & 0.1084 & 0.1093 & 0.1102 & 0.1111 & 0.1120 & 0.1129 & 0.1138 & 0.1147 \\
0.1069 & 0.1079 & 0.1090 & 0.1100 & 0.1111 & 0.1122 & 0.1132 & 0.1143 & 0.1154 \\
0.1062 & 0.1074 & 0.1086 & 0.1098 & 0.1111 & 0.1123 & 0.1136 & 0.1149 & 0.1162 \\
0.1055 & 0.1069 & 0.1082 & 0.1096 & 0.1111 & 0.1125 & 0.1139 & 0.1154 & 0.1169 \\
0.1048 & 0.1063 & 0.1079 & 0.1094 & 0.1110 & 0.1126 & 0.1143 & 0.1159 & 0.1176 \\
0.1041 & 0.1058 & 0.1075 & 0.1093 & 0.1110 & 0.1128 & 0.1146 & 0.1165 & 0.1184
\end{bmatrix}
$$

---

### 5.3 Kết hợp với V (Res sau attention)

$O = V @ A^T$ cho mỗi head, sau đó ghép và reshape về [1,4,3,3].

Kết quả Res (làm tròn đến 4 chữ số):

$$
\text{Res}[:,0,:,:] = 
\begin{bmatrix}
-0.6951 & -0.6948 & -0.6946 \\
-0.6944 & -0.6941 & -0.6939 \\
-0.6937 & -0.6934 & -0.6932
\end{bmatrix},\quad
\text{Res}[:,1,:,:] = 
\begin{bmatrix}
-0.0989 & -0.0986 & -0.0983 \\
-0.0979 & -0.0976 & -0.0973 \\
-0.0969 & -0.0966 & -0.0962
\end{bmatrix},
$$

$$
\text{Res}[:,2,:,:] = 
\begin{bmatrix}
1.6279 & 1.6289 & 1.6298 \\
1.6308 & 1.6318 & 1.6327 \\
1.6337 & 1.6347 & 1.6356
\end{bmatrix},\quad
\text{Res}[:,3,:,:] = 
\begin{bmatrix}
0.4233 & 0.4234 & 0.4235 \\
0.4236 & 0.4237 & 0.4237 \\
0.4238 & 0.4239 & 0.4240
\end{bmatrix}
$$

Có thể thấy, sau attention, các giá trị được tổng hợp toàn cục, với sự chú ý phân bố nhẹ nhàng tăng theo vị trí (do giá trị Q và K âm/dương).

---

### 5.4 Positional Encoding (PE)

Áp dụng Conv3x3 trên Res để thêm thông tin vị trí. Kết quả PE(Res):

$$
\text{PE}[:,0,:,:] = 
\begin{bmatrix}
0.0199 & 0.0239 & 0.0210 \\
0.0280 & 0.0447 & 0.0428 \\
-0.0017 & 0.0037 & 0.0213
\end{bmatrix},\quad
\text{PE}[:,1,:,:] = 
\begin{bmatrix}
-0.0058 & -0.0083 & -0.0089 \\
-0.0070 & -0.0095 & -0.0088 \\
-0.0025 & -0.0024 & -0.0033
\end{bmatrix},
$$

(tương tự cho kênh 2 và 3, với giá trị từ -0.0040 đến 0.1631).

Sau đó, O' = Res + PE(Res).

---

### 5.5 Projection

Áp dụng Conv1x1 trên O' để được output cuối (làm tròn đến 4 chữ số):

$$
\text{Out}[:,0,:,:] = 
\begin{bmatrix}
0.0283 & 0.0288 & 0.0288 \\
0.0271 & 0.0272 & 0.0275 \\
0.0273 & 0.0274 & 0.0273
\end{bmatrix},\quad
\text{Out}[:,1,:,:] = 
\begin{bmatrix}
-0.0003 & -0.0003 & -0.0002 \\
0.0004 & 0.0007 & 0.0006 \\
0.0006 & 0.0009 & 0.0007
\end{bmatrix},
$$

$$
\text{Out}[:,2,:,:] = 
\begin{bmatrix}
-0.0149 & -0.0161 & -0.0163 \\
-0.0156 & -0.0185 & -0.0186 \\
-0.0152 & -0.0172 & -0.0176
\end{bmatrix},\quad
\text{Out}[:,3,:,:] = 
\begin{bmatrix}
-0.0316 & -0.0331 & -0.0335 \\
-0.0326 & -0.0362 & -0.0362 \\
-0.0326 & -0.0352 & -0.0353
\end{bmatrix}
$$

Có thể thấy output đã được làm giàu bởi thông tin toàn cục từ attention, với sự điều chỉnh từ PE và projection.

---

## 6. Ý nghĩa

- **Q, K**: xác định **nên chú ý vào đâu**.
- **V**: chứa **thông tin nội dung**.
- **Softmax(QK^T)**: phân phối xác suất chú ý.
- **PE**: thêm thông tin vị trí.
- **Projection**: hợp nhất các head.

👉 Attention cho phép mỗi pixel **nhìn toàn bộ ảnh** và **tự chọn thông tin** để làm giàu đặc trưng của mình.