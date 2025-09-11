# 🔹 Ví dụ Minh Họa Tính Toán Loss YOLOv11-seg

Dưới đây là ví dụ minh họa tính toán Loss của YOLOv11-seg với dữ liệu giả định nhỏ để dễ theo dõi. Chúng ta sẽ sử dụng dữ liệu cho **batch size B=1**, số anchor points **M=4** (tương ứng 4 đối tượng hoặc vùng), số lớp **C=2** (ví dụ: lớp 0 - background, lớp 1 - object). Cho bounding box, sử dụng **reg_max=4** (để đơn giản hóa DFL, thay vì 16). Cho segmentation, giả định masks là ma trận 2x2 pixel.

**Giả định dữ liệu**:
- **Predicted scores (pred_scores)**: Xác suất dự đoán cho từng anchor thuộc từng lớp (shape [1, 4, 2]).
- **Target scores (target_scores)**: Nhãn ground-truth (shape [1, 4, 2]).
- **Predicted bboxes (pred_bboxes)**: Tọa độ box dự đoán (shape [1, 4, 4]) – (left, top, right, bottom).
- **Target bboxes (target_bboxes)**: Tọa độ box thật (shape [1, 4, 4]).
- **Predicted dist (pred_dist)**: Phân phối cho DFL (shape [1, 4, 4, reg_max=4]) – 4 coords, mỗi coord có 4 bins.
- **Target dist**: Chỉ số left/right cho DFL.
- **Predicted masks (pred_masks)**: Mặt nạ dự đoán sau sigmoid (shape [1, 4, 2x2]).
- **Target masks (M_i)**: Mặt nạ ground-truth (shape [1, 4, 2x2]).
- **Foreground mask (fg_mask)**: Chỉ các anchor positive (giả định tất cả 4 đều positive, sum_fg=4).
- **Target scores sum**: Tổng target_scores = 4 (giả định mỗi anchor có 1 lớp positive).
- **Hyperparameters (hyp)**: λ_box=1.0, λ_seg=1.0, λ_cls=0.5, λ_dfl=1.5 (giả định).

Bây giờ, tính toán từng thành phần Loss theo công thức.

## 1. Tổng Loss

**Công thức tổng quát (từ tài liệu Ultralytics)**:

$$
\mathcal{L}_{total} = B \cdot \Big[ \lambda_{box} \cdot \mathcal{L}_{box} + \lambda_{seg} \cdot \mathcal{L}_{seg} + \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{dfl} \cdot \mathcal{L}_{dfl} \Big]
$$

**Với các hệ số cụ thể:**

- \( B = 1 \)
- \( \lambda_{box} = 1.0 \)
- \( \lambda_{seg} = 1.0 \)
- \( \lambda_{cls} = 0.5 \)
- \( \lambda_{dfl} = 1.5 \)

$$
\mathcal{L}_{total} =
\mathcal{L}_{box} +
\mathcal{L}_{seg} +
0.5 \cdot \mathcal{L}_{cls} +
1.5 \cdot \mathcal{L}_{dfl}
$$

- Chúng ta sẽ tính từng thành phần rồi tổng hợp.

## 2. Classification Loss (Cls Loss)

**Giả định dữ liệu**:
- pred_scores = [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4] (xác suất dự đoán cho 2 lớp).
- target_scores = [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0] (nhãn thật).
- target_scores_sum = max(sum(target_scores), 1) = max(4, 1) = 4.

**Công thức**:

$$
\mathcal{L}_{cls} = \frac{1}{\text{target_scores_sum}} \sum_{i=1}^M \sum_{j=1}^{C} \text{BCE}(p_{ij}, t_{ij})
$$

Trong đó BCE là Binary Cross Entropy:

$$
\text{BCE}(p, t) = - [t \log(p) + (1 - t) \log(1 - p)]
$$

**Tính toán từng bước**:
- Cho anchor 0, lớp 0: BCE(0.9, 1.0) = - [1.0 * log(0.9) + 0.0 * log(0.1)] ≈ - [-0.1054] = 0.1054
- Lớp 1: BCE(0.1, 0.0) = - [0.0 * log(0.1) + 1.0 * log(0.9)] ≈ - [-0.1054] = 0.1054
- Anchor 1: BCE(0.8, 1.0) ≈ 0.2231, BCE(0.2, 0.0) ≈ 0.2231
- Anchor 2: BCE(0.7, 0.0) ≈ 1.2040, BCE(0.3, 1.0) ≈ 1.2040
- Anchor 3: BCE(0.6, 0.0) ≈ 0.9163, BCE(0.4, 1.0) ≈ 0.9163

- Tổng BCE = 0.1054 + 0.1054 + 0.2231 + 0.2231 + 1.2040 + 1.2040 + 0.9163 + 0.9163 ≈ 4.8976
- \mathcal{L}_{cls} = 4.8976 / 4 ≈ 1.2244

**Kết quả**: \mathcal{L}_{cls} ≈ 1.2244 (từ tổng BCE chia cho target_scores_sum=4).

## 3. Bounding Box Loss (Box Loss)

**Giả định dữ liệu**:
- pred_bboxes = [[10, 10, 20, 20], [15, 15, 25, 25], [5, 5, 15, 15], [20, 20, 30, 30]] (left, top, right, bottom).
- target_bboxes = [[10, 10, 20, 20], [10, 10, 20, 20], [5, 5, 15, 15], [20, 20, 30, 30]].
- sum_fg = 4 (tất cả positive).

**Công thức (sử dụng CIoU loss)**:

$$
\mathcal{L}_{box} = \frac{1}{\sum fg} \sum_{i \in fg} (1 - \text{CIoU}(\hat{b}_i, b_i))
$$

CIoU = IoU - (ρ^2 / c^2) - α v, với ρ là khoảng cách trung tâm, c là đường chéo bounding box nhỏ nhất bao cả hai, v là aspect ratio penalty, α là trade-off.

**Tính toán từng bước** (giả định CIoU cho từng box):
- Box 0: CIoU = 1.0 (hoàn toàn khớp) → 1 - 1.0 = 0.0
- Box 1: CIoU ≈ 0.64 (chệch nhẹ) → 1 - 0.64 = 0.36
- Box 2: CIoU = 1.0 → 0.0
- Box 3: CIoU = 1.0 → 0.0

- Tổng (1 - CIoU) = 0.0 + 0.36 + 0.0 + 0.0 = 0.36
- \mathcal{L}_{box} = 0.36 / 4 = 0.09

**Kết quả**: \mathcal{L}_{box} = 0.09 (từ trung bình 1 - CIoU trên 4 positive anchors).

## 4. Distribution Focal Loss (DFL Loss)

**Giả định dữ liệu** (reg_max=4, mỗi coord có phân phối 4 bins):
- pred_dist (cho coord 0 của box 0): [0.1, 0.2, 0.3, 0.4] (softmax để thành prob p_ik).
- target cho coord 0: tl=1, tr=2 (target giữa bin 1 và 2).
- wl = tr - target ≈ 0.5, wr = 1 - wl = 0.5 (giả định target=1.5).
- Tương tự cho các coord/box khác.

**Công thức**:

$$
\mathcal{L}_{dfl} = \frac{1}{\sum fg} \sum_{i \in fg} \sum_{coord=1}^{4} [\text{CE}(p_{i,coord}, tl) \cdot wl + \text{CE}(p_{i,coord}, tr) \cdot wr]
$$

CE(p, t) = - log(p_t), với p là phân phối softmax.

**Tính toán từng bước** (cho 1 coord của 1 box):
- Softmax pred_dist = [0.1, 0.2, 0.3, 0.4] → p = [0.173, 0.211, 0.258, 0.358] (tính softmax).
- CE cho tl=1: - log(p[1]) ≈ - log(0.211) ≈ 1.557
- CE cho tr=2: - log(p[2]) ≈ - log(0.258) ≈ 1.355
- Phần cho coord này: 1.557 * 0.5 + 1.355 * 0.5 ≈ 1.456
- Giả định trung bình cho tất cả coords/boxes: 1.456 * 4 coords * 4 boxes = 23.296
- \mathcal{L}_{dfl} = 23.296 / 4 ≈ 5.824

**Kết quả**: \mathcal{L}_{dfl} ≈ 5.824 (từ trung bình CE weighted trên positive anchors).

## 5. Segmentation Loss (Seg Loss)

**Giả định dữ liệu** (masks 2x2 pixels):
- pred_masks (sau sigmoid, cho anchor 0): [[0.9, 0.8], [0.7, 0.6]]
- target_masks (M_0): [[1, 1], [1, 0]]
- Tương tự cho các anchor khác.

**Công thức**:

$$
\mathcal{L}_{seg} = \frac{1}{\sum fg} \sum_{i \in fg} \Big[ \text{BCE}(\hat{M}_i, M_i) + \text{DiceLoss}(\hat{M}_i, M_i) \Big]
$$

BCE(mask) = trung bình BCE trên pixels.

DiceLoss = 1 - (2 * intersection) / (sum pred + sum target).

**Tính toán từng bước** (cho anchor 0):
- BCE(mask): Trung bình BCE trên 4 pixels ≈ 0.25 (giả định tính).
- DiceLoss: Intersection = 2.1, sum pred=3.0, sum target=3.0 → Dice = 1 - (4.2 / 6.0) ≈ 0.3
- Tổng cho anchor 0: 0.25 + 0.3 = 0.55
- Trung bình cho 4 anchors: 0.55 * 4 = 2.2
- \mathcal{L}_{seg} = 2.2 / 4 = 0.55

**Kết quả**: \mathcal{L}_{seg} = 0.55 (từ trung bình BCE + Dice trên positive anchors).

## 6. Tổng Hợp Loss

**Tính toán**:
- \lambda_{cls} * \mathcal{L}_{cls} = 0.5 * 1.2244 ≈ 0.6122
- \lambda_{box} * \mathcal{L}_{box} = 1.0 * 0.09 = 0.09
- \lambda_{dfl} * \mathcal{L}_{dfl} = 1.5 * 5.824 ≈ 8.736
- \lambda_{seg} * \mathcal{L}_{seg} = 1.0 * 0.55 = 0.55
- Sum weighted = 0.6122 + 0.09 + 8.736 + 0.55 ≈ 9.9882
- \mathcal{L}_{total} = 1 * 9.9882 ≈ 9.9882

**Kết quả**: \mathcal{L}_{total} ≈ 9.9882 (từ tổng các thành phần weighted, nhân B=1). Kết quả này đến từ việc tính trung bình các loss trên positive samples, nhân với lambda, và tổng hợp.