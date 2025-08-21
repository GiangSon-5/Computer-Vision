# Kiến trúc tổng quát của mô hình Object Detection kiểu "One-Stage"  

Trong hầu hết các mô hình phát hiện đối tượng hiện đại (bao gồm cả YOLO, SSD, RetinaNet), kiến trúc được thiết kế theo ba phần chính. Việc chia tách này giúp dễ hiểu hơn vai trò của từng thành phần trong pipeline từ ảnh đầu vào cho đến đầu ra là bounding boxes + nhãn lớp.  

---

## 🟩 Backbone (Trích xuất đặc trưng cơ bản)  

- Nhiệm vụ: biến ảnh gốc thành **feature maps**.  
- Thành phần thường gặp:  
  - Convolution layers  
  - Bottleneck / Residual blocks (giúp học sâu hơn, giảm tham số)  
- Kết quả: feature maps ở nhiều mức độ trừu tượng (low-level edges, high-level semantics).  

> Ví dụ: một ảnh 416×416 khi đi qua Backbone có thể trở thành các feature map 52×52, 26×26, 13×13.  

---

## 🟪 Neck (Kết hợp và khuếch đại đặc trưng)  

- Nhiệm vụ: trộn thông tin từ nhiều tầng feature maps khác nhau.  
- Lý do: đối tượng trong ảnh có thể **rất nhỏ hoặc rất lớn**, nên cần tận dụng cả đặc trưng chi tiết (từ tầng nông) và đặc trưng ngữ nghĩa (từ tầng sâu).  
- Kiến trúc phổ biến:  
  - **FPN (Feature Pyramid Network)**  
  - **PANet (Path Aggregation Network)**  
  - **SPP (Spatial Pyramid Pooling)**  

---

## 🟥 Head (Dự đoán đối tượng)  

- Nhiệm vụ: dự đoán **bounding boxes, confidence score, class label**.  
- Cách hoạt động:  
  - Mỗi cell trên feature map sinh ra một hoặc nhiều dự đoán.  
  - Sử dụng anchor boxes (YOLOv1–YOLOv7, SSD) hoặc anchor-free (YOLOv8, FCOS).  
- Output cuối cùng: danh sách đối tượng với vị trí và nhãn.  

---

## 📌 Sơ đồ kiến trúc tổng quát  

```css
Input Image
   │
   ▼
[ Backbone: Trích xuất đặc trưng ]
   │
   ▼
[ Neck: Kết hợp đa tầng đặc trưng ]
   │
   ▼
[ Head: Phát hiện đối tượng ]
```

## ✅ Tóm tắt

- Backbone: tạo đặc trưng ban đầu từ ảnh.

- Neck: khuếch đại, kết hợp đặc trưng.

- Head: sinh bounding box + lớp đối tượng.

Ba phần này tạo nên “xương sống” chung cho hầu hết các mô hình one-stage object detection.