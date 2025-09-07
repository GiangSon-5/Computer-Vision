# 🔄 Downsampling vs Upsampling trong CNN

## 1️⃣ Downsampling (giảm mẫu, "co ảnh lại")

- **Thường dùng**: `stride=2`, `pooling`, hoặc `conv stride=2`.  
- **Kết quả**: chiều cao (H) và chiều rộng (W) giảm đi một nửa.  
- **Ý nghĩa**: gom thông tin lại, ảnh nhỏ đi nhưng vẫn giữ đặc trưng quan trọng.  

### Ví dụ:
Ảnh gốc: 4 × 4 → Sau downsampling: 2 × 2

```lua
[ 1, 2, 3, 4 ]      [ 2, 4 ]
[ 5, 6, 7, 8 ] →    [ 6, 8 ]
[ 9, 10, 11, 12 ]
[13, 14, 15, 16 ]
```


👉 Ảnh nhỏ hơn, nhưng mỗi pixel sau downsampling đại diện cho một vùng lớn hơn ở ảnh gốc.  

---

## 2️⃣ Upsampling (tăng mẫu, "phóng ảnh ra")

- **Thường dùng**: `nearest neighbor`, `bilinear`, `transpose convolution (deconv)`.  
- **Kết quả**: H và W nhân đôi (ngược lại của downsampling).  
- **Ý nghĩa**: khôi phục lại độ phân giải cao hơn để **dự đoán chi tiết hơn** (thường trong segmentation, GAN).  

### Ví dụ (nearest neighbor):
Ảnh gốc: 2 × 2 → Sau upsampling: 4 × 4  

```lua
[ 1, 2 ]        [ 1, 1, 2, 2 ]
[ 3, 4 ] →      [ 1, 1, 2, 2 ]
                [ 3, 3, 4, 4 ]
                [ 3, 3, 4, 4 ]
```


👉 Ảnh to ra, nhưng không có thêm thông tin mới, chỉ được nội suy/phóng đại.  

---

## ✅ Kết luận
- **Downsampling = chia 2** (giảm H, W).  
- **Upsampling = nhân 2** (tăng H, W).  
- Cả 2 đều **giữ nguyên số kênh (C)**, trừ khi có convolution thay đổi số filters.  
