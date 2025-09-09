

# 🧠 Công thức thu nhỏ Backbone trong YOLO

Trong file `yaml` của YOLO, mỗi block có dạng:

```yaml
[from, repeats, module, args]
```

* **repeats** = số block gốc (Blocks\_base).
* **args** = chứa số kênh gốc (BaseChannel).

Khi scale theo phiên bản (n/s/m/l/x), YOLO dùng 2 siêu tham số:

1. **Chiều sâu (Depth)**

```
Blocks_out = round(Blocks_base × depth_multiple)
```

2. **Chiều rộng (Width)**

```
Channels_out = min(BaseChannel × width_multiple, max_channels)
```

---

# 🧩 Trường hợp YOLOv11-n

Các siêu tham số:

* `depth_multiple = 0.5`
* `width_multiple = 0.25`
* `max_channels = 1024`

---

## Ví dụ với backbone :

```yaml
- [-1, 3, C3k2, [256, False]]          # Block A
- [-1, 3, C3k2, [512, False, 0.25]]    # Block B
```

---

### 🔹 Block A

* Blocks\_base = 3
* BaseChannel = 256

Áp công thức:

```
Blocks_out   = round(3 × 0.5) = 2
Channels_out = min(256 × 0.25, 1024) = min(64, 1024) = 64
```

👉 Block A: **2 block, 64 kênh**

---

### 🔹 Block B

* Blocks\_base = 3
* BaseChannel = 512

Áp công thức:

```
Blocks_out   = round(3 × 0.5) = 2
Channels_out = min(512 × 0.25, 1024) = min(128, 1024) = 128
```

👉 Block B: **2 block, 128 kênh**

---

# ✅ Kết quả Backbone YOLOv11-n

* Block A: 2 block, 64 kênh
* Block B: 2 block, 128 kênh

So với bản “full” (mặc định 3 block, 256–512 kênh), YOLOv11-n đã **giảm một nửa chiều sâu** và **giảm xuống còn 25% số kênh** → mô hình nhẹ và nhanh hơn nhiều.

---

