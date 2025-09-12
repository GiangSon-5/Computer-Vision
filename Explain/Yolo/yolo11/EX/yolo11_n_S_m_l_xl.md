![yolov11](../imgs/yolo11_n_s_m_l_xl.jpg)
# 📊 Ý nghĩa các cột trong bảng cấu hình YOLOv11

## 1. Các cột trong bảng
- **Model variant**: Tên biến thể của mô hình YOLOv11 (`n, s, m, l, xl`).  
  - `n` = nano (rất nhẹ)  
  - `s` = small (nhỏ)  
  - `m` = medium (trung bình)  
  - `l` = large (lớn)  
  - `xl` = extra large (rất lớn)

- **d (depth\_multiple)**: Hệ số nhân cho **số lượng tầng** (layer/block) trong backbone.  
  - Ví dụ: `d = 0.5` → số tầng chỉ còn **một nửa** so với bản gốc.

- **w (width\_multiple)**: Hệ số nhân cho **số lượng kênh (channel)** trong mỗi tầng.  
  - Ví dụ: `w = 0.25` → số kênh giảm xuống **25%** so với bản gốc.

- **mc (max\_channels)**: Giới hạn **số kênh tối đa** trong backbone, không vượt quá giá trị này dù `w` có lớn đến đâu.

---

## 2. Ví dụ minh họa

- **YOLOv11-n**  
  - `depth_multiple = 0.50` → số tầng giảm **một nửa**  
  - `width_multiple = 0.25` → số kênh giảm còn **25%**  
  - `max_channels = 1024` → số kênh tối đa không vượt quá **1024**

- **YOLOv11-xl**  
  - `depth_multiple = 1.00` → số tầng giữ **nguyên bản gốc**  
  - `width_multiple = 1.50` → số kênh tăng lên **150%**  
  - `max_channels = 512` → số kênh tối đa giới hạn ở **512**

---

## 3. Tại sao cần nhiều biến thể?

- **Nhẹ (n, s):** tối ưu cho thiết bị di động, IoT, chạy nhanh với ít tài nguyên.  
- **Mạnh (m, l, xl):** dùng cho server, GPU, cần độ chính xác cao hơn.  

👉 Điều này cho phép người dùng **cân bằng giữa tốc độ và độ chính xác** theo nhu cầu thực tế.

---

## 4. Bảng cấu hình các biến thể YOLOv11

| Model variant | d (depth\_multiple) | w (width\_multiple) | mc (max\_channels) |
|---------------|----------------------|----------------------|---------------------|
| n             | 0.50                 | 0.25                 | 1024                |
| s             | 0.50                 | 0.50                 | 1024                |
| m             | 0.50                 | 1.00                 | 512                 |
| l             | 1.00                 | 1.00                 | 512                 |
| xl            | 1.00                 | 1.50                 | 512                 |

---

# ✅ Tóm tắt

- `depth_multiple (d)` → điều chỉnh **số tầng**.  
- `width_multiple (w)` → điều chỉnh **số kênh**.  
- `max_channels (mc)` → đặt **giới hạn kênh tối đa**.  
- `n, s, m, l, xl` → các biến thể mô hình cho **thiết bị khác nhau**, từ nhẹ đến mạnh.


---
---

# 🔎 Ví dụ minh họa `depth_multiple`, `width_multiple`, `max_channels`

## 1. Mạng gốc (baseline)
Giả sử backbone ban đầu có:
- **4 block** (tầng) → tương ứng với `depth = 4`
- Mỗi block có **[64, 128, 256, 512] kênh**

Biểu diễn đơn giản:

```lua
Block1: 64 kênh
Block2: 128 kênh
Block3: 256 kênh
Block4: 512 kênh
```

---
Ok, mình viết lại cho bạn một bản **Markdown hoàn chỉnh**, có đủ công thức, tính toán và giải thích rõ ràng vai trò của `d (depth_multiple)` chỉ tác động đến block **C3k2** (không ảnh hưởng Conv thường).

---

# YOLOv11 — Ảnh hưởng của `d (depth_multiple)` đến Backbone

## 1. Ý nghĩa của 3 tham số

* **`d (depth_multiple)`**: hệ số nhân cho số lượng block lặp (C3, C2f, C3k2).
  → Không áp dụng cho các Conv đơn lẻ.

* **`w (width_multiple)`**: hệ số nhân cho số lượng kênh (channel).

* **`mc (max_channels)`**: trần trên cho số kênh.

---

## 2. Backbone gốc (baseline)

| Tầng | Kích thước | Kênh gốc | Thành phần  | Block gốc |
| ---- | ---------- | -------- | ----------- | --------- |
| 0    | 640×640    | 3        | Input (ảnh) | -         |
| 1    | 320×320    | 64       | Conv        | -         |
| 2    | 160×160    | 128      | Conv + C3k2 | 3 x d     |
| 3    | 80×80      | 256      | Conv + C3k2 | 6 x d     |
| 4    | 40×40      | 512      | Conv + C3k2 | 6 x d     |
| 5    | 20×20      | 1024     | C3k2        | 3 x d     |

---

## 3. Tính kênh mới với `w = 0.25`

Công thức:

$$
C' = \text{make\_divisible}(C \times w, 8), \quad C' \leq mc
$$

* Tầng 1: $64 × 0.25 = 16$ → 16
* Tầng 2: $128 × 0.25 = 32$ → 32
* Tầng 3: $256 × 0.25 = 64$ → 64
* Tầng 4: $512 × 0.25 = 128$ → 128
* Tầng 5: $1024 × 0.25 = 256$ → 256 (≤ mc=1024)

👉 Kênh mới: **\[16, 32, 64, 128, 256]**

---

## 4. Tính block mới với `d = 0.5`

Công thức:

$$
B' = \max(1, \text{round}(B \times d))
$$

* Tầng 2: $3 × 0.5 = 1.5$ → 2
* Tầng 3: $6 × 0.5 = 3$ → 3
* Tầng 4: $6 × 0.5 = 3$ → 3
* Tầng 5: $3 × 0.5 = 1.5$ → 2

👉 Block mới: **\[2, 3, 3, 2]**

---

## 5. Backbone YOLOv11-n (d=0.5, w=0.25, mc=1024)

| Tầng | Kích thước | Kênh gốc → Kênh mới | Block gốc → Block mới |
| ---- | ---------- | ------------------- | --------------------- |
| 0    | 640×640    | 3 → 3               | -                     |
| 1    | 320×320    | 64 → 16             | -                     |
| 2    | 160×160    | 128 → 32            | 3 → 2                 |
| 3    | 80×80      | 256 → 64            | 6 → 3                 |
| 4    | 40×40      | 512 → 128           | 6 → 3                 |
| 5    | 20×20      | 1024 → 256          | 3 → 2                 |

---

## 6. Giải thích `n = 6 × d`

* Trong paper ghi `n = 6 × d` nghĩa là: số block **C3k2** được điều chỉnh theo `d`.
* Ví dụ tầng 3 gốc có 6 block:

  * YOLOv11-n (`d=0.5`) → $6×0.5=3$ block
  * YOLOv11-s (`d=0.75`) → $6×0.75=4.5$ → 5 block
  * YOLOv11-m/l/xl (`d=1.0`) → $6×1=6$ block

👉 `d` **chỉ tác động đến C3k2 block**, còn Conv đầu vào/giảm kích thước vẫn giữ nguyên.

---

✅ **Kết luận**:
* Backbone YOLOv11-n có kênh giảm còn 25% và block giảm còn một nửa so với bản gốc.




---

## 3. Trường hợp YOLOv11-xl (bị giới hạn max-channel)
- `depth_multiple = 1.0` → số block giữ nguyên **4 block**  
- `width_multiple = 1.5` → số kênh tăng 150%  
- `max_channels = 512` → kênh không vượt quá 512  

Tính toán kênh mới:
- Block1: 64 × 1.5 = 96  
- Block2: 128 × 1.5 = 192  
- Block3: 256 × 1.5 = 384  
- Block4: 512 × 1.5 = 768 nhưng **bị giới hạn bởi max_channels = 512**  

Kết quả:

```lua
Block1: 96 kênh
Block2: 192 kênh
Block3: 384 kênh
Block4: 512 kênh (bị giới hạn bởi max_channels)
```

👉 Backbone vẫn đủ **4 tầng**, nhưng kênh nhiều hơn, mạnh hơn.



---

# ✅ Kết luận
- `depth_multiple (d)` → quyết định **số tầng** (block).  
- `width_multiple (w)` → quyết định **số kênh** trong mỗi tầng.  
- `max_channels (mc)` → đặt **trần giới hạn**, không cho số kênh vượt quá mức này.
