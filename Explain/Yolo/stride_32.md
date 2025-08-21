# 📏 Ảnh đầu vào 640×640, “stride 32” là gì, và liên quan thế nào đến anchor/training/inference?

---

## 1) Vì sao hay dùng **640×640**?
- **Chuẩn bội số của 32** → bảo đảm mọi tầng downsample (chia 2 nhiều lần) đều cho **kích thước nguyên** ở các feature map.  
  Ví dụ: 640 ÷ 32 = **20** (nguyên), nên tầng stride 32 sẽ ra bản đồ **20×20**.  
- **Cân bằng tốc độ–độ chính xác**: 640 đủ lớn để bắt chi tiết, nhưng vẫn chạy nhanh trên GPU phổ biến.  
- Dễ “khớp” với nhiều backbone/neck tiêu chuẩn (FPN/PAFPN) vốn thiết kế theo **stride 8/16/32**.

> Tóm lại: 640 không “thần thánh”, nhưng **rất tiện** cho kiến trúc YOLO đa tỉ lệ (8/16/32).  
> Bạn vẫn có thể dùng 512, 768, 1280… miễn là **bội số của 32**.

---

## 2) “**32**” ở đâu ra? (khái niệm **stride**)
- **Stride (s)** = hệ số thu nhỏ so với ảnh gốc sau nhiều lần downsample.  
- Nhiều backbone giảm kích thước **5 lần liên tiếp ×2** → hệ số tổng là \(2^5 = 32\).  
- Vì vậy ta có các tầng đầu ra thường gặp:
  - **s = 8**  → feature map **80×80** (với input 640)  
  - **s = 16** → feature map **40×40**  
  - **s = 32** → feature map **20×20**

### Chuỗi downsample minh họa (640×640):

```lua
640 → 320 → 160 → 80 → 40 → 20
s=8     s=16    s=32
(80×80) (40×40) (20×20)
```


- Ở mỗi stride **s**, **một cell** trên feature map “nhìn” vào **một ô s×s pixel** trên ảnh gốc.  
  - stride=8 → 1 cell ↔ 8×8 px gốc
  - stride=16 → 1 cell ↔ 16×16 px gốc 
  - stride=32 → 1 cell ↔ 32×32 px gốc

Ví dụ:

- Ảnh gốc = 640×640.

- Feature map stride=32 → size 20×20.

- Nghĩa là 1 cell trên feature map tương ứng với 32×32 pixel vùng ảnh gốc.

---

## 3) Bản đồ đặc trưng khi input **640×640**
| Stride (s) | Kích thước feature map | Mục tiêu chính |
|------------|-------------------------|----------------|
| 8          | 80×80                   | Vật thể nhỏ    |
| 16         | 40×40                   | Vật thể vừa    |
| 32         | 20×20                   | Vật thể lớn    |

> Nhiều phiên bản YOLO dùng **3 đầu ra** (s=8/16/32) để **phát hiện đa tỉ lệ** cùng lúc.

---

## 4) Liên hệ **stride** ↔ **anchor** ↔ **training** ↔ **inference**
- **Anchor**: là “khuôn kích thước” gắn với **mỗi đầu ra** (s=8/16/32).  
  - Tầng **s=32** thường dùng **anchor to hơn** (phù hợp vật thể lớn).  
  - Tầng **s=8** dùng **anchor nhỏ** (vật thể nhỏ).
- **Training (gán nhãn & học)**:  
  1) Xác định cell (trên từng feature map) chứa **tâm** GT box.  
  2) So IoU GT với các **anchor** của tầng đó → chọn anchor tốt nhất.  
  3) Tạo **mục tiêu học (target)**: $t_x, t_y, t_w, t_h$, objectness, class.  
- **Inference (dự đoán)**:  
  - Mô hình xuất **offset** trên từng **cell–anchor**.  
  - **Decode** về toạ độ ảnh bằng **stride s** (để “phóng” từ lưới về pixel).  

---

## 5) Encode & Decode công thức (áp dụng cho input 640, mọi stride)

### Encode (từ GT → target để train)  
Giả sử cell gốc $(c_x, c_y)$, anchor $(p_w, p_h)$, ground-truth box $(g_x, g_y, g_w, g_h)$ (theo pixel).  

$$
t_x = \frac{g_x}{s} - c_x,\quad
t_y = \frac{g_y}{s} - c_y
$$  

$$
t_w = \ln \left(\frac{g_w}{p_w}\right),\quad
t_h = \ln \left(\frac{g_h}{p_h}\right)
$$  

- $(g_x, g_y)$: tâm hộp thật.  
- $(g_w, g_h)$: kích thước hộp thật.  
- Kết quả $(t_x,t_y,t_w,t_h)$ chính là **target** để mô hình học.

---

### Decode (từ output → box dự đoán)  
Giả sử logit dự đoán $(t_x, t_y, t_w, t_h)$:  

$$
b_x = (c_x + \sigma(t_x)) \times s
$$  

$$
b_y = (c_y + \sigma(t_y)) \times s
$$  

$$
b_w = p_w \cdot e^{t_w},\quad
b_h = p_h \cdot e^{t_h}
$$  

- $\sigma(\cdot)$ là sigmoid.  
- $(b_x, b_y)$ là tâm hộp dự đoán (pixel).  
- $(b_w, b_h)$ là kích thước hộp dự đoán (pixel).  

---

## 6) Ví dụ số **nhìn ra “vì sao nhân 32”**
- Input: **640×640**  
- Chọn tầng **s=32** → feature map **20×20**  
- Cell chịu trách nhiệm: $(c_x, c_y) = (8, 5)$  
- Anchor lớn tại tầng này: $(p_w, p_h) = (150, 120)$  
- Dự đoán của mô hình: $t_x=-1.2,\ t_y=-0.7,\ t_w=-0.1,\ t_h=-0.2$  

Tính toán:  

- $\sigma(-1.2) \approx 0.231$, $\sigma(-0.7) \approx 0.332$

- **Tâm hộp**:  

$$
b_x = (8 + 0.231)\times 32 \approx 263\ \text{px}
$$  

$$
b_y = (5 + 0.332)\times 32 \approx 171\ \text{px}
$$  

- **Kích thước**:  

$$
b_w = 150 \cdot e^{-0.1} \approx 135.7\ \text{px},\quad
b_h = 120 \cdot e^{-0.2} \approx 98.2\ \text{px}
$$  

- **Giải thích “×32”**: vì toạ độ đang ở **đơn vị cell** (lưới 20×20), muốn quay về **pixel ảnh** phải **nhân stride s=32**.  

---
---

# Khoảng cách trong k-means cho bounding box

Trong k-means bình thường (trên dữ liệu số), ta hay dùng khoảng cách **Euclidean**:

$$
d((w,h),(w_c,h_c)) = \sqrt{(w - w_c)^2 + (h - h_c)^2}
$$

👉 Nhưng với **bounding box** (hình chữ nhật), khi so sánh thì **hình dạng / tỉ lệ quan trọng hơn kích thước tuyệt đối**.  
Nếu chỉ dùng Euclidean thì 2 box có kích thước gấp đôi nhau (ví dụ `(100,200)` và `(200,400)`) sẽ bị coi là rất xa, trong khi thực ra chúng cùng tỉ lệ và CNN có thể scale được.

---

### Giải pháp trong YOLOv2 (Redmon, 2017)

Thay vì Euclidean, người ta dùng **IoU (Intersection over Union)** để đo độ tương đồng giữa 2 box (giả sử cùng tâm):

$$
IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}
$$

Để biến thành khoảng cách (giá trị càng nhỏ càng tốt), ta lấy:

$$
d = 1 - IoU
$$

→ Đây chính là công thức **“k-means clustering using IoU distance”**.

---

### Các tham số dự đoán trong YOLO

Mỗi bounding box cuối cùng được mô hình hóa qua 4 thông số:

- $t_x$: độ lệch theo **chiều ngang** (tọa độ tâm box so với cell)  
- $t_y$: độ lệch theo **chiều dọc** (tọa độ tâm box so với cell)  
- $t_w$: độ lệch **chiều rộng** so với anchor box  
- $t_h$: độ lệch **chiều cao** so với anchor box  

---

### 🔎 Tóm lại:

- Nếu dùng **Euclidean** → không phù hợp cho bounding box.  
- Nếu dùng **\(1 - IoU\)** → phản ánh trực tiếp mức độ “giống nhau” về hình dạng, **bất kể scale tuyệt đối**.  
- YOLO không dự đoán trực tiếp $(x,y,w,h)$, mà dự đoán các **offsets** $(t_x, t_y, t_w, t_h)$ so với **cell + anchor box**.


# Minh họa *toán tay* — Tìm anchor presets bằng k-means (distance = 1 − IoU)

Dữ liệu ví dụ (các bounding box trong dataset, dạng `(w, h)`):
- B1 = (10, 13)
- B2 = (16, 30)
- B3 = (33, 23)
- B4 = (30, 61)
- B5 = (62, 45)
- B6 = (59, 119)
- B7 = (116, 90)
- B8 = (156, 198)
- B9 = (373, 326)

Chọn số cụm: **k = 3** (muốn 3 anchor presets: nhỏ / trung / lớn).  
Khởi tạo (ví dụ thủ công): các centroid ban đầu
- C1 = (16, 30)  *(nhỏ)*
- C2 = (59, 119) *(trung)*
- C3 = (373, 326) *(lớn)*

---

## Vòng 1 — Gán từng box vào cụm gần nhất (theo IoU)

**Cách tính IoU cho hai hộp (w1,h1) và (w2,h2)** (giả sử cùng tâm):
- Inter = `min(w1,w2) * min(h1,h2)`
- Union = `w1*h1 + w2*h2 - Inter`
- IoU = `Inter / Union`

> Khoảng cách dùng trong k-means = `1 − IoU` (tức càng nhỏ càng giống nếu không dùng 1 trừ thì càng cao càng gần).

Tính nhanh (chỉ nêu Inter, Union, IoU; làm tròn 3 chữ số khi cần):

1. B1 (10×13) với:
   - C1 (16×30): Inter = 10·13 = 130; Union = 130+480−130 = 480 → IoU = 130/480 = **0.271**
   - C2: Inter = 130; Union ≈ 7021 → IoU ≈ **0.0185**
   - C3: Inter = 130; Union ≈ 121598 → IoU ≈ **0.0011**  
   → **Gán C1**

2. B2 (16×30):
   - C1: Inter = 16·30 = 480; Union = 480 → IoU = **1.0**
   → **Gán C1**

3. B3 (33×23):
   - C1: Inter = 16·23 = 368; Union = 759+480−368 = 871 → IoU ≈ **0.423**
   - C2: Inter = 33·23 = 759; Union = 759+7021−759 = 7021 → IoU ≈ **0.108**  
   → **Gán C1**

4. B4 (30×61):
   - C1: Inter = 16·30 = 480; Union = 1830 → IoU ≈ **0.262**
   - C2: Inter = 30·61 = 1830; Union = 7021 → IoU ≈ **0.261**  
   → **Gán C1** (nhỉnh hơn chút)

5. B5 (62×45):
   - C1: Inter = 16·30 = 480; Union = 2790 → IoU ≈ **0.172**
   - C2: Inter = 59·45 = 2655; Union = 2790+7021−2655 = 7156 → IoU ≈ **0.371**  
   → **Gán C2**

6. B6 (59×119):
   - C2: identical → IoU = **1.0**  
   → **Gán C2**

7. B7 (116×90):
   - C2: Inter = 59·90 = 5310; Union = 10440+7021−5310 = 12151 → IoU ≈ **0.437**
   - C3: Inter = 116·90 = 10440; Union = 10440+121598−10440 = 121598 → IoU ≈ **0.086**  
   → **Gán C2**

8. B8 (156×198):
   - C2: Inter = 79·84.666? (xấp xỉ) → IoU ≈ 0.227 (sau tính chi tiết)  
   - C3: Inter = 156·198 = 30888; Union = 30888+121598−30888 = 121598 → IoU ≈ **0.254**  
   → **Gán C3**

9. B9 (373×326):
   - C3: identical → IoU = **1.0**  
   → **Gán C3**

**Kết quả gán (vòng 1):**
- C1: B1, B2, B3, B4  
- C2: B5, B6, B7  
- C3: B8, B9

---

## Vòng 1 — Cập nhật centroids (tính trung bình w,h của từng cụm)

- C1_new = mean w = (10 + 16 + 33 + 30) / 4 = **22.25**  
         mean h = (13 + 30 + 23 + 61) / 4 = **31.75**  
  → C1' = **(22.25, 31.75)**

- C2_new = mean w = (62 + 59 + 116) / 3 = **79.0**  
         mean h = (45 + 119 + 90) / 3 = **84.6667**  
  → C2' = **(79.0, 84.6667)**

- C3_new = mean w = (156 + 373) / 2 = **264.5**  
         mean h = (198 + 326) / 2 = **262.0**  
  → C3' = **(264.5, 262.0)**

---

## Vòng 2 — Gán lại theo centroids mới (tính IoU → gán)

Ta tính lại IoU giữa mỗi box và 3 centroids C1', C2', C3'. (Ở đây mình tóm tắt kết quả chính, chi tiết tính toán giống công thức Inter/Union ở trên.)

Kết quả gán **vòng 2** (sau tính từng IoU thủ công, giống các phép tính trong vòng 1 nhưng với centroids mới):
- C1': B1, B2, B3, B4
- C2': B5, B6, B7
- C3': B8, B9

→ **Không đổi so với vòng 1** → hội tụ.

---

## Kết luận (anchor presets thu được)
Centroids hội tụ sau vài vòng:
- C1 ≈ **(22.25, 31.75)** → làm tròn → **(22, 32)**  *(anchor nhỏ)*  
- C2 ≈ **(79.0, 84.67)** → làm tròn → **(79, 85)**  *(anchor trung)*  
- C3 ≈ **(264.5, 262.0)** → làm tròn → **(264, 262)** *(anchor lớn)*

Đó là cách **tính toán bằng tay** (toán học từng bước) để thấy **anchor presets** xuất phát từ phân bố kích thước GT boxes trong dataset bằng k-means với khoảng cách dựa trên IoU.  

---

### Ghi chú tóm tắt
- Bước chính: dùng IoU làm "độ tương đồng", chạy k-means trên cặp (w,h).  
- Sau hội tụ, lấy centroid (w,h) làm anchor preset (làm tròn nếu cần).  
- Trong thực tế dùng dataset lớn → k thường = 9 (chia đều cho 3 scale), nhưng logic giống nhau.


