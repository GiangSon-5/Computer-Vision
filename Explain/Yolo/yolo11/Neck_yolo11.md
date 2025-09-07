# 🖼️ Deep Learning trong Object Detection: Neck trong YOLO

## 1. Khái niệm Neck
Trong kiến trúc YOLO, *Neck* là phần trung gian, nằm giữa **Backbone** và **Head**.  
Nhiệm vụ chính của Neck là **kết hợp và xử lý đặc trưng đa tầng** trước khi đưa sang Head để dự đoán.  

> Nói cách khác, Backbone giống như "máy chụp ảnh" tạo nhiều bản đồ đặc trưng, còn Neck là "bộ phối trộn thông minh" giúp gộp và tinh chỉnh những đặc trưng này, để Head dễ dàng dự đoán vật thể ở nhiều kích thước khác nhau.

---

## 2. Chức năng chính của Neck
- **Kết hợp đa tỉ lệ (multi-scale fusion)**: kết hợp feature maps ở nhiều độ phân giải → giúp phát hiện vật thể nhỏ, vừa và lớn.  
- **Tăng cường ngữ cảnh (context enhancement)**: tận dụng cả thông tin chi tiết (local) và thông tin toàn cục (global).  
- **Chuẩn bị cho Head**: cung cấp feature map đa dạng, giàu thông tin, phục vụ cho dự đoán bounding box và class.

---

## 3. Thành phần chính trong Neck (YOLO 11)
1. **SPPF (Spatial Pyramid Pooling Fast)**
   - Gom đặc trưng ở nhiều receptive field.
   - Giúp mô hình "nhìn xa trông rộng" mà không thay đổi kích thước ảnh gốc.

2. **C2PSA (Cross-Stage Partial + Position Sensitive Attention)**
   - Cơ chế attention giúp mô hình tập trung vào vùng quan trọng trong ảnh.
   - Học mối quan hệ toàn cục giữa các vị trí đặc trưng.

3. **Upsample (Nâng mẫu)**
   - Tăng kích thước feature map độ phân giải thấp để khớp với feature map có độ phân giải cao hơn.
   - Thường dùng nội suy láng giềng gần nhất (nearest neighbor interpolation).

4. **Concat (Ghép nối)**
   - Ghép các feature maps sau khi upsample với feature maps từ tầng trước.
   - Giữ nguyên chiều cao, chiều rộng nhưng cộng kênh lại → tạo feature map giàu thông tin hơn.

---

## 4. Quy trình hoạt động của Neck
Quá trình xử lý trong Neck thường theo chu trình sau:
1. Feature map từ tầng sâu được đưa qua **SPPF** và/hoặc **C2PSA**.  
2. Thực hiện **Upsample** để tăng độ phân giải.  
3. Thực hiện **Concat** với feature map từ Backbone có cùng độ phân giải.  
4. Lặp lại nhiều lần cho các tầng khác nhau → thu được nhiều feature maps ở các mức độ chi tiết khác nhau.  

> Kết quả: Neck xuất ra các feature maps đa tỉ lệ, chuẩn bị cho Head phát hiện vật thể nhỏ, vừa, lớn.

---

## 5. Sơ đồ ASCII minh họa Neck

> Dữ liệu chảy qua Neck có thể được mô tả như sau:

>     Feature map (low resolution, rich semantics)
>               │
>             SPPF
>               │
>             C2PSA
>               │
>          Upsample ↑
>               │   │
>               └── Concat (with Backbone feature)
>                      │
>                  New feature map
>                      │
>                 → Head (Detect)

---

## 6. Kết luận
Phần **Neck trong YOLO** đóng vai trò:
- Làm cầu nối giữa Backbone và Head.  
- Kết hợp đặc trưng đa tỉ lệ, giữ thông tin chi tiết và ngữ nghĩa.  
- Tăng cường biểu diễn không gian và ngữ cảnh, giúp phát hiện vật thể ở nhiều kích thước hiệu quả hơn.
