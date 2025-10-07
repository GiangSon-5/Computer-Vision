

# 🔹 Ví dụ minh họa PSABlock — Tính toán chi tiết (theo từng bước)


---

## 0) Input sau Positional Encoding (X')

Dữ liệu đầu vào `x` sau khi áp dụng Positional Encoding (PE), được biểu diễn dưới dạng tensor với 4 kênh (`c0'`, `c1'`, `c2'`, `c3'`) cho 4 pixel (`p0`, `p1`, `p2`, `p3`):

| pixel | c0' | c1' | c2' | c3' |
|-------|-----|-----|-----|-----|
| p0    | 1.1 | 0.0 | 1.1 | 2.0 |
| p1    | 2.2 | 1.0 | 0.2 | 2.0 |
| p2    | 3.3 | 0.0 | 1.3 | 2.0 |
| p3    | 4.4 | 1.0 | 0.4 | 2.0 |

> Đây là `x` (sau PE) — đầu vào cho `PSABlock`. Mỗi pixel có 4 chiều (4 kênh), và ta sẽ giả sử đầu ra của PSABlock cũng giữ nguyên 4 kênh (do `c` là số kênh vào/ra, mặc định bằng 4 trong ví dụ này).

---

## 1) Kết quả Attention (đã có)

Kết quả từ `self.attn(x)` được cung cấp sẵn, với mỗi pixel được chiếu về 2 chiều (do `PSABlock` sử dụng attention với kích thước đầu ra giảm xuống, phù hợp với cách thiết kế trong YOLOv11):

| pixel | out_attn[0] | out_attn[1] |
|-------|-------------|-------------|
| p0    | 4.746       | 2.952       |
| p1    | 4.742       | 2.954       |
| p2    | 4.796       | 2.982       |
| p3    | 4.728       | 2.962       |

> `out_attn` là đầu ra của `self.attn(x)`, sử dụng multi-head attention với `num_heads=4` (theo tham số mặc định) và `attn_ratio=0.5`. Kết quả này đã được tính trước, phản ánh sự tập trung vào các mối quan hệ không gian giữa các pixel.

---

## 2) Projection: Chiếu `x` (4-dim) về 2-dim để chuẩn bị cho residual

Giải thích: Để thực hiện residual connection, ta cần ánh xạ `x` (4 chiều) về cùng không gian 2 chiều của `out_attn`. Điều này được thực hiện bằng cách sử dụng một ma trận chiếu `W` (4×2), tương tự ma trận đã dùng trong `self.attn` để tạo Query/Key/Value. Mục đích là tạo một biểu diễn trung gian để cộng với `out_attn` và `ffn_out`.

**Công thức:**

$$
\text{proj}(x) = x \cdot W
$$

Giả sử ma trận `W` (4×2) được định nghĩa như sau (lấy ví dụ để tính tay, tương thích với đầu ra 2 chiều của attention):

```
W = [[1, 0],
     [0, 1],
     [1, 0],
     [0, 1]]
```

Áp dụng cho từng pixel:

- **p0**: `x(p0) = [1.1, 0.0, 1.1, 2.0]`

$$
\text{proj}(x(p0)) = [1.1 \cdot 1 + 0.0 \cdot 0 + 1.1 \cdot 1 + 2.0 \cdot 0, 1.1 \cdot 0 + 0.0 \cdot 1 + 1.1 \cdot 0 + 2.0 \cdot 1] = [2.2, 2.0]
$$

- **p1**: `x(p1) = [2.2, 1.0, 0.2, 2.0]`

$$
\text{proj}(x(p1)) = [2.2 \cdot 1 + 1.0 \cdot 0 + 0.2 \cdot 1 + 2.0 \cdot 0, 2.2 \cdot 0 + 1.0 \cdot 1 + 0.2 \cdot 0 + 2.0 \cdot 1] = [2.4, 3.0]
$$

- **p2**: `x(p2) = [3.3, 0.0, 1.3, 2.0]`

$$
\text{proj}(x(p2)) = [3.3 \cdot 1 + 0.0 \cdot 0 + 1.3 \cdot 1 + 2.0 \cdot 0, 3.3 \cdot 0 + 0.0 \cdot 1 + 1.3 \cdot 0 + 2.0 \cdot 1] = [4.6, 2.0]
$$

- **p3**: `x(p3) = [4.4, 1.0, 0.4, 2.0]`

$$
\text{proj}(x(p3)) = [4.4 \cdot 1 + 1.0 \cdot 0 + 0.4 \cdot 1 + 2.0 \cdot 0, 4.4 \cdot 0 + 1.0 \cdot 1 + 0.4 \cdot 0 + 2.0 \cdot 1] = [4.8, 3.0]
$$

**Bảng `proj(x)` (4→2):**

| pixel | proj(x)[0] | proj(x)[1] |
|-------|------------|------------|
| p0    | 2.200      | 2.000      |
| p1    | 2.400      | 3.000      |
| p2    | 4.600      | 2.000      |
| p3    | 4.800      | 3.000      |

> Lưu ý: Ma trận `W` là giả định để minh họa. Trong thực tế, nó được học từ dữ liệu và có thể khác, nhưng kết quả 2 chiều phù hợp với `out_attn`.

---

## 3) FFN (Feed-Forward Network): Conv 2 → 4 → 2

Giải thích: `self.ffn` bao gồm hai lớp Conv 1x1:
- Lớp đầu (`Conv(c, c * 2, 1)`): Mở rộng từ 2 chiều lên 4 chiều.
- Lớp sau (`Conv(c * 2, c, 1, act=False)`): Nén lại từ 4 chiều về 2 chiều, không dùng activation cuối để giữ tính chất tuyến tính khi cộng residual.

**Trọng số minh họa** (giả định để tính tay):
- `W1` (2×4): Mở rộng từ 2→4
  ```
  W1 = [[2, 0],
        [0, 2],
        [1, 0],
        [0, 1]]
  ```
- `W2` (4×2): Nén từ 4→2
  ```
  W2 = [[1, 0, 1, 0],
        [0, 1, 0, 1]]
  ```

**Công thức:**

$$
h = W_1 \cdot \text{out\_attn}, \quad \text{ffn\_out} = W_2 \cdot h
$$

### 3.1 Tính chi tiết cho p0
- `out_attn(p0) = [4.746, 2.952]`

Tính `h = W1 · out_attn`:

$$
h[0] = 2 \cdot 4.746 + 0 \cdot 2.952 = 9.492
$$

$$
h[1] = 0 \cdot 4.746 + 2 \cdot 2.952 = 5.904
$$

$$
h[2] = 1 \cdot 4.746 + 0 \cdot 2.952 = 4.746
$$

$$
h[3] = 0 \cdot 4.746 + 1 \cdot 2.952 = 2.952
$$

⇒ `h = [9.492, 5.904, 4.746, 2.952]`

Tính `ffn_out = W2 · h`:

$$
ffn\_out[0] = 1 \cdot 9.492 + 0 \cdot 5.904 + 1 \cdot 4.746 + 0 \cdot 2.952 = 9.492 + 4.746 = 14.238
$$

$$
ffn\_out[1] = 0 \cdot 9.492 + 1 \cdot 5.904 + 0 \cdot 4.746 + 1 \cdot 2.952 = 5.904 + 2.952 = 8.856
$$

⇒ **FFN(p0) = [14.238, 8.856]**

### 3.2 Tính cho p1, p2, p3
- **p1**: `out_attn = [4.742, 2.954]`
  - `h = [9.484, 5.908, 4.742, 2.954]`
  - `ffn_out = [9.484 + 4.742, 5.908 + 2.954] = [14.226, 8.862]`

- **p2**: `out_attn = [4.796, 2.982]`
  - `h = [9.592, 5.964, 4.796, 2.982]`
  - `ffn_out = [9.592 + 4.796, 5.964 + 2.982] = [14.388, 8.946]`

- **p3**: `out_attn = [4.728, 2.962]`
  - `h = [9.456, 5.924, 4.728, 2.962]`
  - `ffn_out = [9.456 + 4.728, 5.924 + 2.962] = [14.184, 8.886]`

**Bảng FFN outputs:**

| pixel | ffn_out[0] | ffn_out[1] |
|-------|------------|------------|
| p0    | 14.238     | 8.856      |
| p1    | 14.226     | 8.862      |
| p2    | 14.388     | 8.946      |
| p3    | 14.184     | 8.886      |

---

## 4) Residual (Shortcut) — Cộng lại để ra `out_final`

Giải thích: Vì `self.add = True`, ta thực hiện residual connection bằng cách cộng `proj(x)`, `out_attn`, và `ffn_out`. Mỗi thành phần đều có 2 chiều, phù hợp để tổng hợp.

**Công thức:**

$$
out\_final = \text{proj}(x) + out\_attn + ffn\_out
$$

### Tính từng pixel:
- **p0**:
  - `proj(x) = [2.200, 2.000]`
  - `out_attn = [4.746, 2.952]`
  - `ffn_out = [14.238, 8.856]`

$$
out\_final(p0) = [2.200 + 4.746 + 14.238, 2.000 + 2.952 + 8.856] = [21.184, 13.808]
$$

- **p1**:
  - `proj(x) = [2.400, 3.000]`
  - `out_attn = [4.742, 2.954]`
  - `ffn_out = [14.226, 8.862]`

$$
out\_final(p1) = [2.400 + 4.742 + 14.226, 3.000 + 2.954 + 8.862] = [21.368, 14.816]
$$

- **p2**:
  - `proj(x) = [4.600, 2.000]`
  - `out_attn = [4.796, 2.982]`
  - `ffn_out = [14.388, 8.946]`

$$
out\_final(p2) = [4.600 + 4.796 + 14.388, 2.000 + 2.982 + 8.946] = [23.784, 13.928]
$$

- **p3**:
  - `proj(x) = [4.800, 3.000]`
  - `out_attn = [4.728, 2.962]`
  - `ffn_out = [14.184, 8.886]`

$$
out\_final(p3) = [4.800 + 4.728 + 14.184, 3.000 + 2.962 + 8.886] = [23.712, 14.848]
$$

---

## 5) Bảng kết quả cuối cùng (PSABlock output)

| pixel | out_final[0] | out_final[1] |
|-------|--------------|--------------|
| p0    | 21.184       | 13.808       |
| p1    | 21.368       | 14.816       |
| p2    | 23.784       | 13.928       |
| p3    | 23.712       | 14.848       |

---

## Ghi chú ngắn
- **Projection (`proj(x)`)**: Sử dụng ma trận `W` (4×2) để ánh xạ từ 4 chiều về 2 chiều, phù hợp với `out_attn`. Ma trận này là giả định minh họa, trong thực tế được học từ dữ liệu.
- **FFN**: Trọng số `W1` và `W2` là ví dụ để tính tay; trong code thực tế, chúng được huấn luyện và có thể phức tạp hơn.
- **Residual**: Với `self.add = True`, cả `out_attn` và `ffn_out` đều được cộng với `proj(x)`, tạo ra đầu ra 2 chiều cho mỗi pixel.
- **Trình tự tổng quát**: **Input (4-dim after PE)** → **Projection (4→2)** → **Attention (→out_attn)** → **FFN (2→4→2)** → **Residual sum** → **Output (2-dim per token)**.

---
---

## 1. Projection là gì?

* Trong toán tuyến tính, **projection** (chiếu) là phép nhân ma trận để đưa vector từ một không gian này sang không gian khác.
* Trong deep learning (Attention, FFN, …), “projection” thường dùng để **biến đổi số chiều** của vector đặc trưng.

Ví dụ:

* Input có `dim=4` (vector 4 chiều).
* Ta nhân với ma trận `W (4×2)` → ra vector 2 chiều.
* Đây gọi là **chiếu từ 4D → 2D**.

---

## 2. Projection trong Attention

* Khi tạo Q, K, V, ta có:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

Ở đây:

* `X`: đầu vào (dim=4).
* `W_Q, W_K, W_V`: các ma trận chiếu (projection matrix).
* Kết quả:

  * Q và K thường có `key_dim` nhỏ hơn (ví dụ 2).
  * V thường giữ lại dim gốc.

👉 Nhờ projection mà ta:

* Giảm chiều (đỡ tốn tính toán trong dot-product).
* Tạo không gian biểu diễn riêng cho Q, K, V.

---

## 3. Projection trong Feed-Forward (FFN)

FFN dùng hai phép chiếu:

1. **Expand (projection up):** từ `dim` nhỏ → `dim` lớn (ví dụ 2 → 4).

   * Mở rộng không gian để mạng học biểu diễn phi tuyến phức tạp.
2. **Compress (projection down):** từ `dim` lớn → `dim` gốc (ví dụ 4 → 2).

   * Trả output về cùng kích thước để cộng residual.

👉 Đây chính là lý do FFN mạnh hơn 1 linear đơn: nó “đi một vòng” qua không gian ẩn lớn hơn.

---

## 4. Công thức tổng quát Projection

Với vector $x \in \mathbb{R}^d$ và ma trận $W \in \mathbb{R}^{d \times d'}$:

$$
\text{proj}(x) = xW \quad \in \mathbb{R}^{d'}
$$

* Nếu $d' < d$: **giảm chiều**.
* Nếu $d' > d$: **mở rộng chiều**.

---

✅ Tóm lại:

* **Projection = Linear mapping (chiếu tuyến tính)**.
* Mục tiêu: thay đổi số chiều để tính toán Attention hoặc tạo không gian ẩn trong FFN.

---
