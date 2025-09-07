# 🔹 Ví dụ minh họa PSABlock — tính toán chi tiết (theo từng bước)

> **Ghi chú:** toàn bộ công thức dài đặt trong khối `$$ ... $$` (không render).  
> Công thức mô tả được đặt **sau** phần chữ giải thích và nằm trên dòng mới.

---

## 0) Input sau Positional Encoding (X')

| pixel | c0' | c1' | c2' | c3' |
|-------|-----|-----|-----|-----|
| p0    | 1.1 | 0.0 | 1.1 | 2.0 |
| p1    | 2.2 | 1.0 | 0.2 | 2.0 |
| p2    | 3.3 | 0.0 | 1.3 | 2.0 |
| p3    | 4.4 | 1.0 | 0.4 | 2.0 |

> Đây là $x$ (sau PE) — đầu vào cho PSABlock.

---

## 1) Kết quả Attention (đã có)

| pixel | out_attn[0] | out_attn[1] |
| ----- | ----------- | ----------- |
| p0    | 4.746       | 2.952       |
| p1    | 4.742       | 2.954       |
| p2    | 4.796       | 2.982       |
| p3    | 4.728       | 2.962       |

> `out_attn` là output của `self.attn(x)`.

---

## 2) Projection: chiếu $x$ (4-dim) về 2-dim để cộng residual

#### mở rộng chiều (2 → 4) để tạo không gian ẩn lớn hơn chứa biểu diễn phi tuyến, rồi nén lại (4 → 2) để trả về kích thước ban đầu; giúp FFN có sức biểu diễn mạnh hơn so với một biến đổi tuyến tính đơn thuần.

Giải thích: dùng ma trận chiếu $W$ (4×2) — cùng ma trận đã dùng trước (ví dụ Q/K/V) — để map 4→2.

**Công thức (đặt sau lời giải thích):**

$$
\text{proj}(x) = x \; W
$$

Áp cho từng pixel:

- p0: $x(p0)=[1.1,0.0,1.1,2.0]$  
  $\text{proj}(x(p0))=[1.1+1.1,\;0.0+2.0]=[2.2,\;2.0]$.

- p1: $x(p1)=[2.2,1.0,0.2,2.0]$  
  $\text{proj}(x(p1))=[2.2+0.2,\;1.0+2.0]=[2.4,\;3.0]$.

- p2: $x(p2)=[3.3,0.0,1.3,2.0]$  
  $\text{proj}(x(p2))=[3.3+1.3,\;0.0+2.0]=[4.6,\;2.0]$.

- p3: $x(p3)=[4.4,1.0,0.4,2.0]$  
  $\text{proj}(x(p3))=[4.4+0.4,\;1.0+2.0]=[4.8,\;3.0]$.

**Bảng `proj(x)` (4→2):**

| pixel | proj(x)[0] | proj(x)[1] |
|-------|------------|------------|
| p0    | 2.200      | 2.000      |
| p1    | 2.400      | 3.000      |
| p2    | 4.600      | 2.000      |
| p3    | 4.800      | 3.000      |

---

## 3) FFN (Conv: 2 → 4 → 2) — công thức và trọng số minh họa

Giải thích: `FFN` gồm `W1` (4×2) mở rộng và `W2` (2×4) thu lại. Ở đây ta **giả sử** trọng số để có thể tính tay.

> Trọng số minh họa (ví dụ để tính tay):
> 
> W1 (4×2) =
> >     [2, 0]
> >     [0, 2]
> >     [1, 0]
> >     [0, 1]
> 
> W2 (2×4) =
> >     [1, 0, 1, 0]
> >     [0, 1, 0, 1]

**Công thức (đặt sau lời giải thích):**

$$
h = W_1 \cdot \text{out\_attn}, \qquad
ffn\_out = W_2 \cdot h
$$

### 3.1 Tính chi tiết cho p0

- out_attn(p0) = $[4.746,\;2.952]$.

Tính $h = W_1 \cdot \text{out\_attn}$:

> h[0] = 2*4.746 + 0*2.952 = 9.492  
> h[1] = 0*4.746 + 2*2.952 = 5.904  
> h[2] = 1*4.746 + 0*2.952 = 4.746  
> h[3] = 0*4.746 + 1*2.952 = 2.952

⇒ $h = [9.492,\;5.904,\;4.746,\;2.952]$.

Tính $ffn\_out = W_2 \cdot h$:

> ffn_out[0] = 1*h[0] + 0*h[1] + 1*h[2] + 0*h[3] = 9.492 + 4.746 = 14.238  
> ffn_out[1] = 0*h[0] + 1*h[1] + 0*h[2] + 1*h[3] = 5.904 + 2.952 = 8.856

→ **FFN(p0) = [14.238, 8.856]**.

### 3.2 Tính nhanh cho p1, p2, p3 (cùng W1, W2)

- p1: out_attn = [4.742, 2.954] 
  h = [9.484, 5.908, 4.742, 2.954]  
  FFN(p1) = [9.484+4.742, 5.908+2.954] = **[14.226, 8.862]**

- p2: out_attn = [4.796, 2.982]  
  h = [9.592, 5.964, 4.796, 2.982]  
  FFN(p2) = [9.592+4.796, 5.964+2.982] = **[14.388, 8.946]**

- p3: out_attn = [4.728, 2.962]  
  h = [9.456, 5.924, 4.728, 2.962]  
  FFN(p3) = [9.456+4.728, 5.924+2.962] = **[14.184, 8.886]**

**Bảng FFN outputs:**

| pixel | ffn_out[0] | ffn_out[1] |
|-------|------------|------------|
| p0    | 14.238     | 8.856      |
| p1    | 14.226     | 8.862      |
| p2    | 14.388     | 8.946      |
| p3    | 14.184     | 8.886      |

---

## 4) Residual (shortcut) — cộng lại để ra `out_final`

Giải thích: `self.add = True` bật residual. Ta dùng `proj(x)` (bước 2), `out_attn` (bước 1) và `ffn_out` (bước 3).

**Công thức (dòng mới, không thụt lề):**

$$
out\_final = \text{proj}(x) + out\_attn + ffn\_out
$$

### Tính từng pixel:

- **p0:**

  proj(x) = [2.200, 2.000]  
  out_attn = [4.746, 2.952]  
  ffn_out = [14.238, 8.856]

  $$ out\_final(p0) = [2.200,2.000] + [4.746,2.952] + [14.238,8.856] = [21.184,\;13.808] $$

- **p1:**

  proj(x) = [2.400, 3.000]  
  out_attn = [4.742, 2.954]  
  ffn_out = [14.226, 8.862]

  $$ out\_final(p1) = [2.400,3.000] + [4.742,2.954] + [14.226,8.862] = [21.368,\;14.816] $$

- **p2:**

  proj(x) = [4.600, 2.000]  
  out_attn = [4.796, 2.982]  
  ffn_out = [14.388, 8.946]

  $$ out\_final(p2) = [4.600,2.000] + [4.796,2.982] + [14.388,8.946] = [23.784,\;13.928] $$

- **p3:**

  proj(x) = [4.800, 3.000]  
  out_attn = [4.728, 2.962]  
  ffn_out = [14.184, 8.886]

  $$ out\_final(p3) = [4.800,3.000] + [4.728,2.962] + [14.184,8.886] = [23.712,\;14.848] $$

---

## 5) Bảng kết quả cuối cùng (PSABlock output)

| pixel | out_final[0] | out_final[1] |
|-------|--------------:|--------------:|
| p0    | 21.184        | 13.808        |
| p1    | 21.368        | 14.816        |
| p2    | 23.784        | 13.928        |
| p3    | 23.712        | 14.848        |

---

## Ghi chú ngắn

- `proj(x)` được lấy bằng cùng ma trận chiếu $W$ đã dùng trong phần Q/K/V để chuyển 4→2; đây là cách minh hoạ để residual có cùng kích thước với `out_attn`.  
- $W_1, W_2$ ở FFN là **ví dụ** để tính tay; code thật học các trọng số khác.  
- Toàn bộ trình tự: **Input (4-dim after PE)** → **Projection (4→2)** → **Attention (→out_attn)** → **FFN (2→4→2)** → **Residual sum** → **Output (2-dim per token)**.
