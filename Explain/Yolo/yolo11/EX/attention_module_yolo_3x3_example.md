
> Input ban đầu chỉ có **4 kênh**, sao sau `Conv1×1` lại ra tới **8 kênh** (shape `(1, 8, 3, 3)`)?

---

### Giải thích

1. **Input**: `(B=1, C=4, H=3, W=3)`

   * Có 4 kênh đặc trưng.

2. **Conv1×1 để tạo QKV**:
   Trong `__init__` bạn thấy đoạn này:

```python
nh_kd = self.key_dim * num_heads   # tổng kích thước của tất cả K hoặc Q
h = dim + nh_kd * 2                # số kênh output = V + Q + K
self.qkv = Conv(dim, h, 1, act=False)
```

   Với giá trị bạn chọn:

   * `dim = 4`
   * `num_heads = 2`
   * `head_dim = 2`
   * `key_dim = 1`
   * `nh_kd = key_dim * num_heads = 1 * 2 = 2`

   → `h = dim + nh_kd * 2 = 4 + 2*2 = 8`

   ✅ Nghĩa là `qkv` sẽ có **8 kênh output**.

3. **Ý nghĩa**:

   * Trong 8 kênh này:

     * **2 kênh cho Q (Query)** (vì mỗi head có `key_dim=1`, 2 head → tổng 2 kênh)
     * **2 kênh cho K (Key)** (tương tự)
     * **4 kênh cho V (Value)** (bằng với `dim` gốc = 4)

   Tổng cộng: **2 + 2 + 4 = 8 kênh**.
   → Đây là lý do bạn thấy tensor `QKV` có shape `(1, 8, 3, 3)`.
---

1. **Tính kích thước của từng head**:

   ```python
   self.head_dim = dim // num_heads
   ```

   Với `dim=4`, `num_heads=2` → `head_dim = 4 // 2 = 2`.

2. **Khi reshape qkv** trong `forward`:

   ```python
   q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)...
   ```

3. **`key_dim = int(head_dim * attn_ratio)`**  
    ```python
    self.key_dim = int(self.head_dim * attn_ratio) # kích thước của Query/Key
    ```
→ quyết định kích thước Q, K.

---

###  Với ví dụ 

* `dim = 4` (tổng số kênh đầu vào).

* `num_heads = 2`.

* `head_dim = 4 // 2 = 2`.
  → Mỗi head sẽ có **2 kênh cho Value (V)**.

* `attn_ratio = 0.5`.

* `key_dim = int(2 * 0.5) = 1`.
  → Mỗi head sẽ có **1 kênh cho Query** và **1 kênh cho Key**.

---

### 3. Tổng cộng trên tất cả heads

* Query (Q): `key_dim * num_heads = 1 * 2 = 2` kênh.
* Key (K): `key_dim * num_heads = 1 * 2 = 2` kênh.
* Value (V): `head_dim * num_heads = 2 * 2 = 4` kênh.

Tổng = `2 + 2 + 4 = 8` kênh → đúng với `h = dim + nh_kd * 2`.

---

### 4. Vì sao Q, K nhỏ hơn V?

* Trong thực tế, **V giữ lại nhiều thông tin gốc** (để truyền về sau).
* **Q và K chỉ dùng để tính toán attention weights** → không cần quá nhiều chiều.
* Do đó người ta đặt một tỉ lệ `attn_ratio < 1` để **giảm số chiều của Q, K**, tiết kiệm tính toán.

---

