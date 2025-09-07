
# 🔎 ASCII Flow minh họa

```lua
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

```
> Nói cách khác:
> Cell center = “vị trí gốc”
> Anchor = “khuôn mẫu kích thước”
> Offset = “chỉnh sửa” → tạo bounding box cuối cùng
> -> không sinh box trực tiếp, mà dùng anchor làm nền, rồi dự đoán độ lệch để ra box.
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