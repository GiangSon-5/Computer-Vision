# 🖼️ Deep Learning trong Computer Vision: Khái niệm Anchor

## 1. Vấn đề trong Object Detection
Trong *computer vision*, một bài toán quan trọng là **Object Detection** (nhận diện vật thể).  
Mục tiêu: xác định *vị trí* và *loại* của đối tượng trong ảnh.  

- Đầu ra không chỉ là nhãn (*label*) mà còn là **hộp giới hạn** (*bounding box*).  
- Mỗi bounding box được mô tả bằng tọa độ `(x, y, w, h)`.  

> Vấn đề: ảnh có thể chứa nhiều đối tượng với kích thước, tỉ lệ khác nhau → mô hình cần cách **dự đoán đa dạng bounding box**.

---

## 2. Khái niệm Anchor Box
**Anchor box** là các *hộp tham chiếu* (reference boxes) được định nghĩa trước với nhiều kích thước và tỉ lệ khác nhau.  
Mỗi anchor đóng vai trò như "khung" để mô hình dự đoán xem có vật thể nào khớp với nó không.  

- Một ảnh được chia thành nhiều **grid cells**.  
- Tại mỗi cell, ta gán nhiều anchor boxes có hình dạng khác nhau (vuông, ngang, dọc).  
- Mô hình **không dự đoán hộp từ đầu**, mà chỉ **tinh chỉnh (offset)** các anchor để khớp với vật thể thật.  

👉 Ví dụ anchor box:  
- Anchor 1: hình vuông **50×50**  
- Anchor 2: hình chữ nhật ngang **100×50**  
- Anchor 3: hình chữ nhật dọc **50×100**  


# 🛠 Pipeline YOLO — từ ảnh → box kết quả 

# 🔗 **[Minh họa](../Yolo/cell_anchorbox_ASCII.md)**

---

## 🔎 Bước 0. Input  
- Ảnh đầu vào (ví dụ: 416×416)  
- Ground truth (hộp đỏ) = nhãn do con người gán  

---

## 🟦 Bước 1. Chia lưới (Grid)  
- Ảnh được backbone thu nhỏ thành feature map, ví dụ 13×13.  
- Mỗi ô (cell) trong lưới phụ trách vùng 32×32 px trên ảnh gốc.  
- Tâm object nằm trong cell nào → cell đó phụ trách object đó.  

👉 Ví dụ: tâm 🐕 nằm trong cell (8,5).  

---

## 📐 Bước 2. Anchor box. Mô hình: đã có sẵn anchor templates (ví dụ: 150×120, 40×30, 300×250, …).
- Trong mỗi cell, có nhiều anchor box (khuôn mẫu kích thước khác nhau).  
- Ta tính IoU giữa GT box và các anchor.  
- Anchor có IoU cao nhất sẽ được gán cho object.  

👉 Ví dụ: anchor (150×120) hợp với hộp chó (140×100).  
> Mục đích của Anchor Box:

>> - Cung cấp các “khuôn tham chiếu” để mô hình không phải dự đoán bounding box từ số 0.

>> - Giúp bao phủ nhiều tỉ lệ và kích thước khác nhau (vuông, ngang, dọc).

>> - Tăng khả năng phát hiện đa dạng đối tượng trong ảnh.
---

## 🧮 Bước 3. Mô hình dự đoán  
Ở mỗi cell–anchor, mô hình xuất ra:  
- Độ lệch tâm trong cell: t_x, t_y  
- Độ lệch kích thước so với anchor: t_w, t_h  
- Objectness (có object hay không)  
- Class scores (thuộc loại gì: chó, mèo…)  

👉 Mô hình **không dự đoán trực tiếp box**, mà dự đoán các giá trị lệch “offset” này.  

---

## 🔄 Bước 4. Encode (GT → target t)  
- Từ hộp thật (GT), ta đổi sang dạng (t_x, t_y, t_w, t_h) để mô hình học.  

Ý tưởng:  
- t_x, t_y = vị trí tâm object trong cell  
- t_w, t_h = tỉ lệ kích thước object so với anchor  

👉 Đây là giá trị “đúng” mà mô hình cần tiệm cận.  

---

## 📤 Bước 5. Decode (t → box dự đoán)  
- Khi chạy dự đoán, mô hình cho ra (t_x, t_y, t_w, t_h).  
- Dùng công thức sigmoid + exp để biến ngược lại thành box trên ảnh:  

$$
b_x = (c_x + \sigma(t_x)) \times \text{stride}  
$$  

$$
b_y = (c_y + \sigma(t_y)) \times \text{stride}  
$$  

$$
b_w = p_w \cdot e^{t_w}, \quad b_h = p_h \cdot e^{t_h}  
$$  

👉 Nếu mô hình học tốt → box vàng ≈ box đỏ.  

---

## 🗑 Bước 6. NMS (lọc trùng)  
- Các cell lân cận cũng có thể “nhảy vào dự đoán”, tạo ra nhiều box cho cùng một object.  
- Non-Max Suppression sẽ giữ lại box có score cao nhất, loại bớt box trùng lặp.  

👉 Kết quả cuối cùng chỉ còn 1 box vàng.  

---

## ✅ Bước 7. Output  
Danh sách các object dạng:  
(bbox, score, class)  

Ví dụ:  
```python
(x=260, y=170, w=140, h=100, score=0.9, class=dog)
```


---

# 🌟 Tóm gọn pipeline
1. Chia ảnh thành grid cells  
2. Trong cell chứa object → chọn anchor box tốt nhất  
3. Mô hình dự đoán offset (t_x, t_y, t_w, t_h)  
4. Decode offset → box dự đoán  
5. NMS → lọc trùng, giữ box tốt nhất  

👉 Như vậy dễ thấy:  
- **Cell** → xác định vùng chịu trách nhiệm (theo tâm object)  
- **Anchor** → khuôn kích thước ban đầu  
- **Offset (t_x, t_y, t_w, t_h)** → mô hình học cách “biến anchor thành hộp thật”  

---
---

# VD:

# 🚶‍♂️ Object Detection từng bước (ví dụ YOLO) — từ ảnh → grid → anchor → dự đoán → decode → NMS

> **Quy ước màu trong hình minh họa**  
> **Đỏ** = Ground Truth bounding box (hộp thật do người gán nhãn)  
> **Vàng** = Bounding box mô hình dự đoán


## Minh họa Grid (ví dụ thu nhỏ)

> Mỗi số là chỉ số cột (c_x).  
> Hàng là chỉ số dòng (c_y).  
> 🔴 = cell chứa tâm object (chó).  

```lua
|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | . | . | . | . | . | . | . | . | . | . |
| 1 | . | . | . | . | . | . | . | . | . | . |
| 2 | . | . | . | . | . | . | . | . | . | . |
| 3 | . | . | . | . | . | . | . | . | . | . |
| 4 | . | . | . | . | . | . | . | . | . | . |
| 5 | . | . | . | . | . | . | . | . | 🔴 | . |
| 6 | . | . | . | . | . | . | . | . | . | . |
```

---

## Bước 0 — Thiết lập ví dụ
- Ảnh đầu vào: **416×416**  
- Ta xét 1 đối tượng duy nhất (ví dụ: **chó**) với **Ground Truth (màu đỏ)**:
  - Tâm: **(x_gt, y_gt) = (260 px, 170 px)**
  - Kích thước: **(w_gt, h_gt) = (140 px, 100 px)**

### 🔍 Tọa độ các cạnh của bounding box:

- **Trái**:  
  $x_{\text{left}} = x_{\text{gt}} - \frac{w_{\text{gt}}}{2} = 260 - \frac{140}{2} = 190$

- **Phải**:  
  $x_{\text{right}} = x_{\text{gt}} + \frac{w_{\text{gt}}}{2} = 260 + \frac{140}{2} = 330$

- **Trên**:  
  $y_{\text{top}} = y_{\text{gt}} - \frac{h_{\text{gt}}}{2} = 170 - \frac{100}{2} = 120$

- **Dưới**:  
  $y_{\text{bottom}} = y_{\text{gt}} + \frac{h_{\text{gt}}}{2} = 170 + \frac{100}{2} = 220$

> Bounding box có góc trên trái tại $(190, 120)$ và góc dưới phải tại $(330, 220)$.

---

## Bước 1 — Ảnh → Feature maps (giữ spatial info)
Backbone + Neck tạo ra nhiều feature map. Ở ví dụ này ta dùng **feature map 13×13** (stride = 32 px/cell):
- Mỗi **cell** đại diện cho vùng **32×32 px** trên ảnh gốc.
- Tọa độ ô lưới chứa **tâm** của object:
  - **c_x = floor(260 / 32) = 8**
  - **c_y = floor(170 / 32) = 5**

> Cell chịu trách nhiệm chính cho object là **(c_x, c_y) = (8, 5)**.

---

## Bước 2 — Chọn Anchor Box phù hợp
Giả sử ta có 3 anchor cho scale 13×13:
- **A1 = (100, 80)**, **A2 = (150, 120)**, **A3 = (60, 40)**

Tính IoU (giả sử các hộp cùng tâm để minh họa đơn giản):

- Với **A1 (100×80)**:
  - Diện tích giao: **min(140,100) × min(100,80) = 100 × 80 = 8000**
  - Diện tích hợp: **140×100 + 100×80 − 8000 = 14000**
  - **IoU = 8000 / 14000 ≈ 0.571**

- Với **A2 (150×120)**:
  - Giao: **min(140,150) × min(100,120) = 140 × 100 = 14000**
  - Hợp: **140×100 + 150×120 − 14000 = 18000**
  - **IoU = 14000 / 18000 ≈ 0.777** ✅ *tốt nhất*

- Với **A3 (60×40)**:
  - **IoU nhỏ ≈ 0.17**

→ **Chọn anchor tốt nhất: A2 = (p_w, p_h) = (150, 120)**.

---

## Bước 3 — Mô hình dự đoán trên cell (8,5)
Tại mỗi anchor trong cell (8,5), mô hình **không dự đoán trực tiếp** hộp thật mà dự đoán các **tham số lệch**:

- **Box Coordinates**: $$t_x, t_y, t_w, t_h$$  
- **Objectness**: $$p_o$$  
- **Class scores**: $$p_1, p_2, \dots, p_c$$

> Ý tưởng: **t_x, t_y** là độ lệch tâm *trong ô*, còn **t_w, t_h** là độ lệch kích thước *so với anchor*.

---

## Bước 4 — Encode trực giác (từ GT[Ground Truth box] → t[prediction target])
Để hiểu decode dễ hơn, ta trước hết xem **nếu mô hình “hoàn hảo”** thì các tham số **t** sẽ ra sao.

- Độ lệch tâm *trong ô (8,5)*:
  - Gốc ô theo pixel: **(256, 160)**
  - $$\Delta_x = \frac{260 - 256}{32} = 0.125,\quad \Delta_y = \frac{170 - 160}{32} = 0.3125$$
  - Encode (vì $\sigma(t_x) = \Delta_x,\ \sigma(t_y) = \Delta_y$):
    - $$t_x = \mathrm{logit}(0.125) \approx -1.946$$
    - $$t_y = \mathrm{logit}(0.3125) \approx -0.789$$

- Độ lệch kích thước so với anchor **(150,120)**:
  - $$t_w = \ln\!\left(\frac{w_{gt}}{p_w}\right) = \ln\!\left(\frac{140}{150}\right) \approx -0.069$$
  - $$t_h = \ln\!\left(\frac{h_{gt}}{p_h}\right) = \ln\!\left(\frac{100}{120}\right) \approx -0.182$$

> Đây là các giá trị *mục tiêu* để mô hình học.

---

## Bước 5 — Decode (từ t → hộp dự đoán màu vàng)
Khi suy luận, mô hình xuất ra $$t_x, t_y, t_w, t_h$$. Ta **decode** về tọa độ thực:

$$
b_x = \sigma(t_x) + c_x
$$

$$
b_y = \sigma(t_y) + c_y
$$

$$
b_w = p_w \cdot e^{t_w}
$$

$$
b_h = p_h \cdot e^{t_h}
$$

Giả sử mô hình dự đoán gần với “lý tưởng” ở trên:
- $$t_x = -1.946,\ t_y = -0.789,\ t_w = -0.069,\ t_h = -0.182$$

Ta có:
- $$b_x = \sigma(-1.946) + 8 \approx 0.125 + 8 = 8.125$$
- $$b_y = \sigma(-0.789) + 5 \approx 0.312 + 5 = 5.312$$
- $$b_w = 150 \cdot e^{-0.069} \approx 140\ \text{px}$$
- $$b_h = 120 \cdot e^{-0.182} \approx 100\ \text{px}$$

**Quy đổi về pixel (stride 32 px):**
- **Tâm**: **(8.125×32, 5.312×32) ≈ (260 px, 170 px)**  
- **Kích thước**: **(140 px, 100 px)**

> Kết quả: **hộp vàng** trùng gần như hoàn hảo với **hộp đỏ (ground truth)**.

---

## Bước 6 — Các cell lân cận & NMS
- Các cell lân cận *cũng có thể dự đoán* hộp cho cùng một object (objectness thấp hơn).
- Sau khi thu tất cả dự đoán từ **13×13, 26×26, 52×52** (mỗi cell 3 anchors), ta áp dụng **Non-Max Suppression (NMS)**:
  1. Sắp xếp các box theo **score = p_o × p_{class}**
  2. Lấy box score cao nhất, **loại** các box trùng lặp có **IoU > ngưỡng** (ví dụ 0.5)
  3. Lặp lại đến khi hết box

> Nhờ **NMS**, ta **không “gom nhiều cell thành 1 box”**, mà **giữ lại box tốt nhất** cho mỗi đối tượng.

---

## Bước 7 — Kết quả cuối cùng
- Đầu ra: danh sách **(bbox, score, class)** cho mỗi đối tượng.  
- Trong ví dụ: mô hình trả về **1 box vàng** khớp hộp đỏ (chó), cùng với **objectness** cao và **class “dog”**.

---

## Tóm tắt ngắn gọn quy trình
1. **Grid**: ảnh → lưới (vd: 13×13), chọn cell chứa **tâm** object.  
2. **Anchor**: chọn **anchor** có IoU cao nhất với object.  
3. **Dự đoán**: mô hình trả ra $$t_x, t_y, t_w, t_h, p_o, \{p_c\}$$ cho từng anchor/cell.  
4. **Decode**: dùng $$\sigma, \exp$$ và **(c_x,c_y), (p_w,p_h)** để ra **(b_x,b_y,b_w,b_h)**.  
5. **NMS**: lọc trùng, giữ box tốt nhất.

---

## Ghi chú (dễ nhầm)
- **Mỗi object chỉ gán cho 1 cell chính** (cell chứa tâm), **không** phải gom từ nhiều cell.  
- Nếu **anchor nhỏ hơn** vật thể thật, mô hình sẽ học **tăng** $$t_w, t_h$$ (qua hàm mũ) để **phóng to** hộp.  
- Nhiều **feature map** (13/26/52) giúp phát hiện **đa tỉ lệ** (nhỏ–vừa–lớn).

---
