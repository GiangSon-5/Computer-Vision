# 📊 Minh họa H × W × C trong CNN qua ví dụ quả banh 🎾

## 1. Ảnh gốc (Input RGB)

- Kích thước: **640 × 640 × 3**
  - H = 640: chiều cao
  - W = 640: chiều rộng
  - C = 3: số kênh (RGB)

👉 Mỗi pixel chỉ có 3 giá trị màu.  
Ví dụ pixel ở giữa (320, 320):

```lua
[123, 45, 200] # R=123, G=45, B=200
```

---

## 2. Minh họa bằng ảnh nhỏ 5×5 RGB

Ảnh chứa một quả banh (trắng) trên nền xanh:

```lua
[ [ [0,255,0], [0,255,0], [255,255,255], [0,255,0], [0,255,0] ],
[ [0,255,0], [255,255,255], [255,255,255], [255,255,255], [0,255,0] ],
[ [0,255,0], [255,255,255], [255,255,255], [255,255,255], [0,255,0] ],
[ [0,255,0], [0,255,0], [255,255,255], [0,255,0], [0,255,0] ],
[ [0,255,0], [0,255,0], [0,255,0], [0,255,0], [0,255,0] ] ]
```

- `[0,255,0] = xanh (nền)`
- `[255,255,255] = trắng (quả banh)`

---

## 3. Qua Convolution stride=2

- Kích thước giảm còn: **320 × 320 × 64**
  - H, W: giảm một nửa
  - C: tăng từ 3 → 64

👉 Pixel không còn là `[R,G,B]` nữa, mà thành **vector 64 đặc trưng**.

Ví dụ pixel (160, 160):

```lua
[0.12, -1.3, 2.5, 0.0, ..., 1.8] # vector dài 64
```

---

## 4. Ý nghĩa của 64 kênh đặc trưng

- Trong RGB: 3 số = 3 màu.
- Trong feature map: 64 số = 64 cách nhìn khác nhau về ảnh.

Ví dụ vài filters (bộ lọc):

- Kênh 1: phát hiện cạnh dọc
- Kênh 2: phát hiện cạnh ngang
- Kênh 3: phát hiện góc
- Kênh 10: nhận diện texture tròn
- ...
- Kênh 64: đặc trưng phức tạp hơn

---

## 5. Demo feature maps từ quả banh

🎯 **Filter 1 (cạnh dọc):**

```lua
[ [0,0,1,0,0],
[0,1,1,1,0],
[0,1,1,1,0],
[0,0,1,0,0],
[0,0,0,0,0] ]
```


🎯 **Filter 2 (cạnh ngang):**

```lua
[ [0,0,0,0,0],
[0,0,1,0,0],
[0,1,1,1,0],
[0,0,1,0,0],
[0,0,0,0,0] ]
```


🎯 **Filter 3 (hình tròn):**

```lua
[ [0,0,1,0,0],
[0,1,1,1,0],
[1,1,1,1,1],
[0,1,1,1,0],
[0,0,1,0,0] ]
```


👉 Mỗi pixel trong feature map = "mức độ khớp" với đặc trưng.

---

## ✅ Kết luận

- **H, W**: giảm dần qua pooling/stride (ảnh nhỏ lại).  
- **C (channels)**: tăng dần (từ 3 màu → hàng chục, hàng trăm đặc trưng).  
- Pixel trong feature map không còn là "màu sắc", mà là **vector đặc trưng** mô tả cạnh, hình tròn, góc, texture…  
- Nhờ đó CNN mới hiểu được hình dạng quả banh chứ không chỉ thấy màu trắng trên nền xanh.

