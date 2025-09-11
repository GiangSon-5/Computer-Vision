# 🔹 Ví dụ Minh Họa Tính Toán Loss YOLOv11-seg

Chúng ta sử dụng dữ liệu giả định nhỏ để dễ theo dõi:

- **Batch size**: $B = 1$
- **Số anchor points**: $M = 4$ (tương ứng 4 đối tượng hoặc vùng)
- **Số lớp**: $C = 2$ (ví dụ: lớp 0 - background, lớp 1 - object)
- **Bounding box**: sử dụng $reg\_max = 4$ (đơn giản hóa DFL, thay vì 16)
- **Segmentation**: giả định masks là ma trận $2 \times 2$ pixel

### 🎯 Giả định dữ liệu

- **Predicted scores (`pred_scores`)**  
  Xác suất dự đoán cho từng anchor thuộc từng lớp  
  `shape = [1, 4, 2]`  
  → Batch size = 1, có 4 anchor boxes, mỗi anchor dự đoán xác suất cho 2 lớp (background, object)

- **Target scores (`target_scores`)**  
  Nhãn ground-truth cho từng anchor  
  `shape = [1, 4, 2]`  
  → Mỗi anchor có nhãn one-hot cho 2 lớp (ví dụ: `[1, 0]` nếu là background)

- **Predicted bboxes (`pred_bboxes`)**  
  Tọa độ box dự đoán (left, top, right, bottom)  
  `shape = [1, 4, 4]`  
  → Mỗi anchor có 4 giá trị tọa độ: `[x1, y1, x2, y2]`

- **Target bboxes (`target_bboxes`)**  
  Tọa độ box ground-truth  
  `shape = [1, 4, 4]`  
  → Cấu trúc giống `pred_bboxes`, dùng để tính loss định vị

- **Predicted dist (`pred_dist`)**  
  Phân phối xác suất cho từng tọa độ box theo DFL  
  `shape = [1, 4, 4, reg_max=4]`  
  → Với mỗi anchor, mỗi tọa độ (4 coords) có phân phối xác suất trên 4 bins (softmax)

- **Target dist**  
  Chỉ số bin trái/phải gần nhất với giá trị tọa độ thật  
  → Dùng để tính loss DFL bằng cách nội suy giữa 2 bin gần nhất

- **Predicted masks (`pred_masks`)**  
  Mặt nạ phân đoạn dự đoán sau sigmoid  
  `shape = [1, 4, 2x2]`  
  → Mỗi anchor có mặt nạ kích thước 2×2 pixel, giá trị từ 0–1

- **Target masks (`M_i`)**  
  Mặt nạ ground-truth  
  `shape = [1, 4, 2x2]`  
  → Mỗi anchor có mặt nạ nhị phân (0 hoặc 1) để so sánh với `pred_masks`

- **Foreground mask (`fg_mask`)**  
  Mặt nạ đánh dấu các anchor positive (được dùng để tính loss)  
  → Giả định tất cả 4 anchor đều positive → `sum_fg = 4`

- **Target scores sum**  
  Tổng số nhãn dương trong batch  
  → `target_scores_sum = 4` (mỗi anchor có 1 lớp positive)

- **Hyperparameters (`hyp`)**  
  Các hệ số điều chỉnh trọng số cho từng thành phần loss:  
  - $\lambda_{box} = 1.0$ → trọng số cho Box Loss  
  - $\lambda_{seg} = 1.0$ → trọng số cho Segmentation Loss  
  - $\lambda_{cls} = 0.5$ → trọng số cho Classification Loss  
  - $\lambda_{dfl} = 1.5$ → trọng số cho Distribution Focal Loss


Bây giờ, tính toán từng thành phần Loss theo công thức.

## 1. Tổng Loss

**Công thức tổng quát (từ tài liệu Ultralytics)**:

$$
\mathcal{L}_{total} = B \cdot \Big[ \lambda_{box} \cdot \mathcal{L}_{box} + \lambda_{seg} \cdot \mathcal{L}_{seg} + \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{dfl} \cdot \mathcal{L}_{dfl} \Big]
$$

**Với các hệ số cụ thể:**

- $B = 1$
- $\lambda_{box} = 1.0$
- $\lambda_{seg} = 1.0$
- $\lambda_{cls} = 0.5$
- $\lambda_{dfl} = 1.5$

$$
\mathcal{L}_{total} =
\mathcal{L}_{box} +
\mathcal{L}_{seg} +
0.5 \cdot \mathcal{L}_{cls} +
1.5 \cdot \mathcal{L}_{dfl}
$$

- Chúng ta sẽ tính từng thành phần rồi tổng hợp.

## 2. 📊 Classification Loss (Cls Loss)

### 🔧 Giả định dữ liệu

- **Predicted scores (`pred_scores`)**:  
  `[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]`  
  → Xác suất dự đoán cho 2 lớp (background, object)

- **Target scores (`target_scores`)**:  
  `[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]`  
  → Nhãn ground-truth

- **Tổng nhãn dương (`target_scores_sum`)**:  
  `max(sum(target_scores), 1) = max(4, 1) = 4`

---

### 📐 Công thức

$$
\mathcal{L}_{cls} = 
\frac{1}{\text{target}\_{\text{scores}\_{\text{sum}}}}
\sum_{i=1}^{M} \sum_{j=1}^{C} \text{BCE}(p_{ij}, t_{ij})
$$


Trong đó Binary Cross Entropy (BCE) được tính như sau:

$$
\text{BCE}(p, t) = - [t \cdot \log(p) + (1 - t) \cdot \log(1 - p)]
$$

---

### 🧮 Tính toán từng bước

| Anchor | Lớp | $p$ | $t$ | BCE(p, t) |
|--------|-----|-----|-----|------------|
| 0      | 0   | 0.9 | 1.0 | 0.1054     |
| 0      | 1   | 0.1 | 0.0 | 0.1054     |
| 1      | 0   | 0.8 | 1.0 | 0.2231     |
| 1      | 1   | 0.2 | 0.0 | 0.2231     |
| 2      | 0   | 0.7 | 0.0 | 1.2040     |
| 2      | 1   | 0.3 | 1.0 | 1.2040     |
| 3      | 0   | 0.6 | 0.0 | 0.9163     |
| 3      | 1   | 0.4 | 1.0 | 0.9163     |

**Tổng BCE**:  

$$
\text{Total BCE} = 0.1054 + 0.1054 + 0.2231 + 0.2231 + 1.2040 + 1.2040 + 0.9163 + 0.9163 \approx 4.8976
$$

**Loss phân loại**:  

$$
\mathcal{L}_{cls} = \frac{4.8976}{4} \approx 1.2244
$$

---

### ✅ Kết quả

$$
\mathcal{L}_{cls} \approx 1.2244
$$

→ Đây là giá trị trung bình BCE trên tất cả các anchor và lớp, được chia theo tổng nhãn dương.
## 3. 📦 Bounding Box Loss (Box Loss)

### 🔧 Giả định dữ liệu

- **Predicted bboxes (`pred_bboxes`)**:  
  `[[10, 10, 20, 20], [15, 15, 25, 25], [5, 5, 15, 15], [20, 20, 30, 30]]`  
  → Tọa độ box dự đoán (left, top, right, bottom)

- **Target bboxes (`target_bboxes`)**:  
  `[[10, 10, 20, 20], [10, 10, 20, 20], [5, 5, 15, 15], [20, 20, 30, 30]]`  
  → Tọa độ box ground-truth

- **Foreground anchors (`sum_fg`)**:  
  `4` (giả định tất cả đều positive)

---

### 📐 Công thức CIoU Loss

$$
\mathcal{L}_{box} = \frac{1}{\sum fg} \sum_{i \in fg} \left(1 - \text{CIoU}(\hat{b}_i, b_i)\right)
$$

Trong đó:

- $\hat{b}_i$: box dự đoán  
- $b_i$: box ground-truth  
- CIoU = IoU − $\frac{\rho^2}{c^2}$ − $\alpha v$  
  - $\rho$: khoảng cách giữa tâm hai box  
  - $c$: đường chéo của bounding box nhỏ nhất bao cả hai  
  - $v$: độ lệch tỉ lệ khung hình  
  - $\alpha$: hệ số điều chỉnh

---

### 🧮 Tính toán từng bước (giả định CIoU)

| Box | CIoU | $1 - \text{CIoU}$ |
|-----|------|-------------------|
| 0   | 1.00 | 0.00              |
| 1   | 0.64 | 0.36              |
| 2   | 1.00 | 0.00              |
| 3   | 1.00 | 0.00              |

**Tổng loss**: 

$$
\sum (1 - \text{CIoU}) = 0.00 + 0.36 + 0.00 + 0.00 = 0.36
$$

**Loss trung bình**:  

$$
\mathcal{L}_{box} = \frac{0.36}{4} = 0.09
$$

---

### ✅ Kết quả

$$
\mathcal{L}_{box} = 0.09
$$

→ Đây là giá trị trung bình của $1 - \text{CIoU}$ trên 4 anchor positive.
## 4. 🎯 Distribution Focal Loss (DFL Loss)

### 🔧 Giả định dữ liệu

- **reg_max = 4** → mỗi tọa độ (coord) có phân phối 4 bins
- **pred_dist** (cho coord 0 của box 0): `[0.1, 0.2, 0.3, 0.4]`  
  → sẽ được chuẩn hóa bằng softmax thành xác suất `p_ik`
- **target** cho coord 0: `tl = 1`, `tr = 2` (target nằm giữa bin 1 và 2)
- **Trọng số**:  
  - $wl = tr - target = 0.5$  
  - $wr = 1 - wl = 0.5$  
  → giả định target = 1.5

---

### 📐 Công thức

$$
\mathcal{L}_{dfl} = \frac{1}{\sum fg} \sum_{i \in fg} \sum_{coord=1}^{4} \left[
\text{CE}(p_{i,coord}, tl) \cdot wl + \text{CE}(p_{i,coord}, tr) \cdot wr
\right]
$$

Trong đó:

- $\text{CE}(p, t) = -\log(p_t)$  
  → Cross Entropy tại bin mục tiêu $t$  
- $p$ là phân phối xác suất sau softmax

---

### 🧮 Tính toán từng bước (cho 1 coord của 1 box)

- **Softmax** của `[0.1, 0.2, 0.3, 0.4]` →  
  $p = [0.173, 0.211, 0.258, 0.358]$

- **CE cho tl = 1**:  
  $-\log(0.211) \approx 1.557$

- **CE cho tr = 2**:  
  $-\log(0.258) \approx 1.355$

- **Weighted CE cho coord này**:  
  $1.557 \cdot 0.5 + 1.355 \cdot 0.5 \approx 1.456$

- **Giả định trung bình cho tất cả coords và boxes**:  
  $1.456 \cdot 4 \text{ coords} \cdot 4 \text{ boxes} = 23.296$

- **Loss trung bình**:  
  $$
  \mathcal{L}_{dfl} = \frac{23.296}{4} \approx 5.824
  $$

---

### ✅ Kết quả

$$
\mathcal{L}_{dfl} \approx 5.824
$$

→ Đây là giá trị trung bình của Cross Entropy có trọng số trên các anchor positive.
## 5. 🧩 Segmentation Loss (Seg Loss)

### 🔧 Giả định dữ liệu

- **pred_masks** (sau sigmoid, cho anchor 0):  
  `[[0.9, 0.8], [0.7, 0.6]]`

- **target_masks (`M_0`)**:  
  `[[1, 1], [1, 0]]`

- Tương tự cho các anchor khác (giả định tất cả đều positive)

---

### 📐 Công thức

$$
\mathcal{L}_{seg} = \frac{1}{\sum fg} \sum_{i \in fg} \left[
\text{BCE}(\hat{M}_i, M_i) + \text{DiceLoss}(\hat{M}_i, M_i)
\right]
$$

Trong đó:

- **BCE(mask)**: trung bình Binary Cross Entropy trên các pixel  
- **DiceLoss**:  
  $$
  \text{DiceLoss} = 1 - \frac{2 \cdot \text{intersection}}{\text{sum pred} + \text{sum target}}
  $$

---

### 🧮 Tính toán từng bước (cho anchor 0)

- **BCE(mask)**:  
  Trung bình BCE trên 4 pixels ≈ `0.25` (giả định)

- **DiceLoss**:  
  - Intersection = `2.1`  
  - Sum pred = `3.0`  
  - Sum target = `3.0`  
  - Dice = $1 - \frac{4.2}{6.0} \approx 0.3$

- **Tổng loss cho anchor 0**:  
  `0.25 + 0.3 = 0.55`

- **Trung bình cho 4 anchors**:  
  `0.55 × 4 = 2.2`

- **Loss trung bình**:  
  $$
  \mathcal{L}_{seg} = \frac{2.2}{4} = 0.55
  $$

---

### ✅ Kết quả

$$
\mathcal{L}_{seg} = 0.55
$$

→ Đây là giá trị trung bình của BCE và DiceLoss trên các anchor positive.
## 6. 📊 Tổng Hợp Loss

### 🧮 Tính toán từng thành phần

- $\lambda_{cls} \cdot \mathcal{L}_{cls} = 0.5 \cdot 1.2244 \approx 0.6122$
- $\lambda_{box} \cdot \mathcal{L}_{box} = 1.0 \cdot 0.09 = 0.09$
- $\lambda_{dfl} \cdot \mathcal{L}_{dfl} = 1.5 \cdot 5.824 \approx 8.736$
- $\lambda_{seg} \cdot \mathcal{L}_{seg} = 1.0 \cdot 0.55 = 0.55$

**Tổng weighted loss**:  

$$
\text{Sum weighted} = 0.6122 + 0.09 + 8.736 + 0.55 \approx 9.9882
$$

**Loss tổng thể**: 

$$
\mathcal{L}_{total} = B \cdot \text{Sum weighted} = 1 \cdot 9.9882 \approx 9.9882
$$

---

### ✅ Kết quả

$$
\mathcal{L}_{total} \approx 9.9882
$$

→ Đây là tổng loss sau khi nhân từng thành phần với hệ số $\lambda$ tương ứng và lấy trung bình trên các mẫu positive. Kết quả phản ánh mức độ sai lệch tổng thể của mô hình trên batch giả định.
