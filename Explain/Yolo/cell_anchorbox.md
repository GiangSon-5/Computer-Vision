# 🛠 Pipeline YOLO — từ ảnh → box kết quả

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

---

## 🧮 Bước 3. Mô hình dự đoán  
Ở mỗi cell–anchor, mô hình xuất ra:  
- Độ lệch tâm trong cell: t_x, t_y  
- Độ lệch kích thước so với anchor: t_w, t_h  
- Objectness (có object hay không)  
- Class scores (thuộc loại gì: chó, mèo…)  

👉 Mô hình **không dự đoán trực tiếp box**, mà dự đoán các “offset” này.  

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

# 🔎 ASCII Flow minh họa
📥 ẢNH GỐC (416x416)  
    ↓  
🧠 Backbone + Neck  
    ↓  
🟩 FEATURE MAP (13x13)  
    ↓  
Cell (8,5) chứa tâm 🐕 → chọn Anchor (150×120)  
    ↓  
📊 Dự đoán offset (t_x, t_y, t_w, t_h, p_o, p_class)  
    ↓  
📤 Decode → Box dự đoán (vàng) ≈ Box GT (đỏ)  
    ↓  
🗑 NMS → loại trùng  
    ↓  
✅ OUTPUT: (bbox, score, class)  


---
---


# Cell–Anchor trong YOLO

## 1. Cell là gì?  
- Sau backbone, ảnh được thu nhỏ thành **feature map** (ví dụ 13×13).  
- Mỗi **cell** trong 13×13 này tương ứng với 1 vùng ảnh gốc (khoảng 32×32 px).  
- Nếu **tâm object** rơi vào cell nào → cell đó **chịu trách nhiệm** cho object đó.  

---

## 2. Anchor là gì?  
- Trong mỗi cell, YOLO đặt sẵn **k nhiều anchor box** (kích thước khác nhau).  
- Các anchor này đều có **tâm tại đúng tâm cell**, nhưng **kích thước khác nhau**.  
- Ví dụ ở cell (8,5), có 3 anchor:  
  - (40×30)  
  - (150×120)  
  - (300×250)  

---

## 3. Ghép lại: Cell–Anchor  
- “**Cell–Anchor**” nghĩa là: một **anchor cụ thể tại một cell cụ thể**.  
- Nếu có **13×13 cells** và mỗi cell có **3 anchors** → tổng cộng **13×13×3 “cell–anchor”**.  
- Mỗi cell–anchor là một **“ứng viên” hộp dự đoán**.  

---

## 4. Pipeline dễ hiểu hơn  
- **Cell:** chọn vùng chịu trách nhiệm (theo tâm object).  
- **Anchor:** trong cell đó, chọn khuôn gần đúng nhất (theo IoU).  
- **Cell–Anchor:** chính là “khuôn mẫu tại cell đó” mà mô hình sẽ tinh chỉnh bằng offset → thành box cuối cùng.  

---

## 📌 Ví dụ cụ thể  
- Con chó có **tâm rơi vào cell (8,5)**.  
- Ở cell (8,5) có 3 anchor:  
  - Anchor A: 40×30  
  - Anchor B: 150×120 ✅ (IoU cao nhất với GT box 140×100)  
  - Anchor C: 300×250  
- Khi training:  
  - **Cell (8,5), Anchor B** được “giao trách nhiệm”.  
  - Mô hình dự đoán offset để biến **Anchor B → hộp thật** của chó.  

