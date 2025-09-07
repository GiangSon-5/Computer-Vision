# 🧩 C2 PSA trong YOLO 11  
*(Cross Stage Partial with Position Sensitive Attention)*

## 1. Tổng quan
- **C2 PSA** là một **khối mới** được giới thiệu trong YOLO 11.  
- Viết tắt của **Cross Stage Partial with Position Sensitive Attention**.  
- Vai trò chính: **học mối quan hệ toàn cục giữa các pixel/đặc trưng ở nhiều vị trí → nâng cao khả năng biểu diễn không gian**.  
- Xuất hiện trong **Neck** – nơi YOLO kết hợp đặc trưng từ backbone để chuẩn bị cho phát hiện đối tượng ở head.

---

## 2. Vị trí trong kiến trúc YOLO 11
YOLO có 3 phần: **Backbone – Neck – Head**.  
- **Backbone:** trích xuất đặc trưng ban đầu.  
- **Neck:** kết hợp và làm giàu đặc trưng.  
- **Head:** dự đoán hộp + nhãn.  

Trong Neck:  
# 🧩 C2 PSA trong YOLO 11  
*(Cross Stage Partial with Position Sensitive Attention)*

## 1. Tổng quan
- **C2 PSA** là một **khối mới** được giới thiệu trong YOLO 11.  
- Viết tắt của **Cross Stage Partial with Position Sensitive Attention**.  
- Vai trò chính: **học mối quan hệ toàn cục giữa các pixel/đặc trưng ở nhiều vị trí → nâng cao khả năng biểu diễn không gian**.  
- Xuất hiện trong **Neck** – nơi YOLO kết hợp đặc trưng từ backbone để chuẩn bị cho phát hiện đối tượng ở head.

---

## 2. Vị trí trong kiến trúc YOLO 11
YOLO có 3 phần: **Backbone – Neck – Head**.  
- **Backbone:** trích xuất đặc trưng ban đầu.  
- **Neck:** kết hợp và làm giàu đặc trưng.  
- **Head:** dự đoán hộp + nhãn.  

Trong Neck:  

```lua
Backbone → SPF (Spatial Pyramid Pooling Fast) → C2 PSA → UpSample + Concat với C3 K2 → ...
```


- SPF tạo biểu diễn cố định cho nhiều kích thước đối tượng.  
- Sau đó **C2 PSA** xử lý để mô hình hiểu rõ **bối cảnh không gian** trước khi đưa đi concat/upsample.  

---

## 3. Cấu tạo & Tham số
Trong file cấu hình YOLO, module C2 PSA có dạng:  



```lua
[from, repeats, module, args]
```


- **from:** kết nối từ block trước đó.  
- **repeats:** xác định số lượng khối PSA bên trong.  
  - Tham số `n = repeats × depth_multiple`.  
  - `depth_multiple` thay đổi theo biến thể YOLO (Nano, Small, Medium, Large, XLarge).  
- **module:** chính là `C2PSA`.  
- **args:** các đối số, trong đó quan trọng nhất là **base output channel**.  

### 🔹 Base Output Channel
- Là số kênh đầu ra cơ sở của C2 PSA.  
- Cách tính kênh đầu ra cuối cùng:  

```lua
out_channels = min(base_output_channel, max_channels) × width_multiple
```

- Tham số này điều chỉnh độ “rộng” của đặc trưng và cho phép YOLO 11 linh hoạt theo biến thể (nhẹ → mạnh).  

---

## 4. Tương tác với các khối khác
- **UpSample:** sau khi qua C2 PSA, feature map được **phóng to bằng nearest neighbor upsampling** để khớp với C3 K2.  
- **Concat:** ghép đặc trưng từ C2 PSA với C3 K2.  
- Ví dụ:  
  - C3 K2 đầu ra: `40×40×512`  
  - C2 PSA sau upsample: `40×40×512`  
  - Concat: `40×40×1024`  
- **Kết nối tiếp theo:** sau concat, đặc trưng đi qua convolutional block và các concat khác.  
- **Head (Detect Block):**  
- Dù C2 PSA không trực tiếp nối vào Detect Block, nhưng đặc trưng nó sinh ra sẽ đi qua nhiều concat với C3 K2.  
- Các detect block sau đó dùng để phát hiện **vật lớn – trung bình – nhỏ**.

---

## 5. Ý nghĩa & Lợi ích
- **Mới trong YOLO 11:** lần đầu xuất hiện, thay thế cho các block cũ trong Neck.  
- **Nâng cao biểu diễn không gian:** không chỉ học đặc trưng cục bộ mà còn học quan hệ toàn cục → hiểu ngữ cảnh ảnh tốt hơn.  
- **Linh hoạt theo biến thể:** thông qua `repeats × depth_multiple` (độ sâu) và `base output channel` (độ rộng).  
- **Tăng độ chính xác:** đặc biệt với:  
- Vật thể nhỏ trong ảnh lớn.  
- Vật thể bị che khuất.  
- Cảnh nhiều chi tiết phức tạp.  

---

## 6. Sơ đồ luồng dữ liệu minh họa

```lua
Ảnh gốc
│
▼
Backbone
│
▼
SPF (Spatial Pyramid Pooling Fast)
│
▼
C2 PSA ──► UpSample ──► Concat với C3 K2 ──► Convolution ──► ... ──► Head (Detect)

```


---

# 🎯 Tóm tắt nhanh
- **C2 PSA = block mới trong Neck YOLO 11.**  
- **Chức năng:** học quan hệ toàn cục, tăng cường biểu diễn không gian.  
- **Tham số chính:**  
  - `repeats × depth_multiple` → số PSA blocks.  
  - `base output channel` → kênh đầu ra cơ sở.  
- **Vai trò:** tạo đặc trưng mạnh, giàu ngữ cảnh → hỗ trợ detect chính xác hơn.

----
