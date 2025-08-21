# 🔎 CNN truyền thống (phiên bản cơ bản)

- Pipeline thường thấy:

```css
Ảnh đầu vào
   │
[Conv + ReLU]  → trích xuất đặc trưng cục bộ
   │
[Pooling]      → giảm kích thước (downsampling)
   │
[Flatten]      → biến tensor thành vector
   │
[Fully Connected] → phân loại (ví dụ 10 lớp MNIST)
   │
[Softmax]

```

> Đây là kiến trúc như LeNet-5, AlexNet, VGG.

> Output là vector class probabilities.

> Ứng dụng: image classification (phân loại ảnh).

# 🔎 CNN hiện đại trong Object Detection (YOLO, Faster R-CNN…)

- Pipeline đã thay đổi:

```csharp
Ảnh đầu vào
   │
[Backbone: Conv + Bottleneck/Residual blocks]
   │
[Neck: FPN/PANet/SPP → kết hợp nhiều feature maps]
   │
[Head: Prediction layers → bbox + confidence + class]

```

## 🔎 CNN cổ điển (classification)

- Giả sử ảnh đầu vào 4×4:

```less
Input (4×4):
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 1, 2, 3],
 [4, 5, 6, 7]]

```
- Qua Conv + Pooling → giảm kích thước, cuối cùng Flatten thành *vector 1D*:

```csharp
Flatten:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7]
```

> Sau đó đưa vào Fully Connected (FC) để ra class probabilities (ví dụ 10 lớp).

> → Mất luôn thông tin không gian (spatial info: vị trí gốc pixel nào ở đâu).


## 🔎 CNN hiện đại (detection – YOLO, Faster R-CNN …)

- Vẫn ảnh đầu vào 4×4. Sau Backbone (Conv nhiều tầng + Bottleneck), ta vẫn giữ feature map 2D:

```lua
Feature map (giả sử còn 2×2, 3 kênh):
Channel 1:
[[0.2, 0.7],
 [0.5, 0.9]]

Channel 2:
[[0.1, 0.4],
 [0.3, 0.8]]

Channel 3:
[[0.9, 0.2],
 [0.6, 0.3]]

```
- Đây không Flatten nữa, mà giữ nguyên tensor 2D (2×2×3).

- Mỗi cell (ô lưới) trong feature map sẽ dự đoán:

  - Bounding box (x, y, w, h)

  - Confidence score

  - Class label

- Ví dụ:

  - Ô [0,0] dự đoán có một chiếc xe hơi ở góc trên trái ảnh.

  - Ô [1,1] dự đoán có một người ở góc dưới phải.

---
## 📌 So sánh trực quan

- CNN cổ điển:

```lua
Ảnh → Conv/Pooling → Flatten → FC → Class
(chỉ biết "ảnh này là con mèo", không biết mèo ở đâu)
```

CNN hiện đại (YOLO):

```lua
Ảnh → Conv/Bottleneck → Feature Maps (2D giữ spatial info) → Head
(biết "ảnh có con mèo, tọa độ (x,y,w,h)")
```

## ✅ Kết luận

- Flatten + FC (cũ): phù hợp classification, nhưng mất thông tin vị trí.

- Giữ feature map 2D (mới): phù hợp detection, segmentation, vì vẫn còn spatial info để dự đoán đối tượng ở đâu.

---
---
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