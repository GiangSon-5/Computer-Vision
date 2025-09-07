# 🔹 Minh họa Q/K/V với tính toán chi tiết

## 1️⃣ Ma trận đầu vào

Giả sử ma trận đầu vào gồm 4 pixel, mỗi pixel có 4 kênh đặc trưng:

| pixel | c0 | c1 | c2 | c3 |
|-------|----|----|----|----|
| p0    | 1  | 0  | 1  | 2 |
| p1    | 2  | 1  | 0  | 2 |
| p2    | 3  | 0  | 1  | 2 |
| p3    | 4  | 1  | 0  | 2 |

- N = 4 pixel  
- C = 4 kênh  

> Mỗi pixel là một "token", mỗi cột là kênh đặc trưng.

---

## 2️⃣ Xây dựng Q/K/V bằng ma trận linear Wq/Wk/Wv

### Wq (Query) – chọn kênh c0 và c2

> ```
> Wq = [[1,0],  # c0 -> Q chiều 0
>       [0,0],  # c1 không dùng
>       [0,1],  # c2 -> Q chiều 1
>       [0,0]]  # c3 không dùng
> ```

Tính Q cho từng pixel:

> - p0: `[1,0,1,2] · Wq` → `[1*1+0*0+1*0+2*0, 1*0+0*0+1*1+2*0] = [1,1]`  
> - p1: `[2,1,0,2] · Wq` → `[2,0]`  
> - p2: `[3,0,1,2] · Wq` → `[3,1]`  
> - p3: `[4,1,0,2] · Wq` → `[4,0]`

```lua
Q = [[1,1],
    [2,0],
    [3,1],
    [4,0]]
```


---

### Wk (Key) – chọn kênh c0 và c1 (trộn vị trí)

> ```
> Wk = [[0,1],  # c0 -> K chiều 1
>       [1,0],  # c1 -> K chiều 0
>       [0,0],  # c2 không dùng
>       [0,0]]  # c3 không dùng
> ```

Tính K cho từng pixel:

> - p0: `[1,0,1,2] · Wk` → `[0*1+0*0+0*0+0*0, 1*1+0*0+1*0+2*0] = [0,1]`  
> - p1: `[2,1,0,2] · Wk` → `[1,2]`  
> - p2: `[3,0,1,2] · Wk` → `[0,3]`  
> - p3: `[4,1,0,2] · Wk` → `[1,4]`


---

### Wk (Key) – chọn kênh c0 và c1 (trộn vị trí)

> ```
> Wk = [[0,1],  # c0 -> K chiều 1
>       [1,0],  # c1 -> K chiều 0
>       [0,0],  # c2 không dùng
>       [0,0]]  # c3 không dùng
> ```

Tính K cho từng pixel:

> - p0: `[1,0,1,2] · Wk` → `[0*1+0*0+0*0+0*0, 1*1+0*0+1*0+2*0] = [0,1]`  
> - p1: `[2,1,0,2] · Wk` → `[1,2]`  
> - p2: `[3,0,1,2] · Wk` → `[0,3]`  
> - p3: `[4,1,0,2] · Wk` → `[1,4]`

```lua
K = [[0,1],
    [1,2],
    [0,3],
    [1,4]]
```


---

### Wv (Value) – chọn kênh c1 và c2

> ```
> Wv = [[0,0],
>       [0,1],  # c1 -> V chiều 1
>       [1,0],  # c2 -> V chiều 0
>       [0,0]]
> ```

Tính V cho từng pixel:

> - p0: `[1,0,1,2] · Wv` → `[1,0]`  
> - p1: `[2,1,0,2] · Wv` → `[0,1]`  
> - p2: `[3,0,1,2] · Wv` → `[1,0]`  
> - p3: `[4,1,0,2] · Wv` → `[0,1]`

```lua
V = [[1,0],   # p0
     [0,1],   # p1
     [1,0],   # p2
     [0,1]]   # p3


```
# Giải thích các tham số 

## 1️⃣ B = batch size = 1
- Vì giả sử chỉ có 1 hình ảnh hoặc 1 batch input, nên **B = 1**.  
- Nếu có nhiều hình ảnh cùng lúc thì **B** sẽ là số lượng hình ảnh trong batch.

## 2️⃣ num_heads = 1
- Đây là số attention heads.  
- Multi-head attention chia vector Q/K/V thành nhiều phần (head) để học các hướng quan hệ khác nhau.  
- Ở ví dụ này mình lấy **1 head** cho đơn giản → chỉ có 1 head duy nhất.

## 3️⃣ key_dim = 2
- `key_dim` là số chiều của vector Q và K mỗi head.  
- Ví dụ chúng ta muốn Q và K mỗi pixel là vector 2 chiều → **key_dim = 2**.

## 4️⃣ head_dim = 2
- `head_dim` là số chiều của vector V mỗi head.  
- Trong một số kiến trúc, V có thể có chiều khác K, ở đây mình lấy **2** → **head_dim = 2**.

## 5️⃣ N = 4 pixel
- `N = H*W` = tổng số pixel khi flatten chiều height/width.  
- Input có 2×2 pixel → **N = 2*2 = 4**.


---

## 3️⃣ Gộp Q/K/V trong code `self.qkv(x)`

> Trong code Attention:
>
> - `self.qkv(x)` là Conv1x1, **tương đương linear projection** gộp Wq/Wk/Wv
> - Output shape `(B, dim+2*nh_kd, H, W)`  
> - `.view(B, num_heads, key_dim*2+head_dim, N).split(...)` → tách ra Q/K/V

> Ở ví dụ này:
>
> - B = 1, num_heads = 1, key_dim = 2, head_dim = 2, N = 4 pixel  
> - Sau split:  
>     - Q: `(1,1,2,4)` = giá trị vừa tính  
>     - K: `(1,1,2,4)`  
>     - V: `(1,1,2,4)`

---

## 4️⃣ Ý nghĩa

- **Mỗi pixel có vector Q/K/V riêng**: dùng cho dot-product attention  
- **Q/K/V này chính là linear projection**: `x · Wq/Wk/Wv` gộp trong Conv1x1  
- Tiếp theo sẽ tính **attention scores**:  

```lua
attn = softmax(Q^T @ K / sqrt(key_dim))
out = V @ attn^T
```


- Vector output là **tổng hợp giá trị V được weighted bởi attention**.

