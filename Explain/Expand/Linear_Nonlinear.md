## So sánh Tuyến tính vs Phi tuyến (Activation Function)

Giả sử sau Convolution ta thu được feature map:

```lua
Z = [
[-2, -1],
[ 1, 2]
]
```


---

### 1. Không dùng Activation (Tuyến tính)

Giữ nguyên:

```lua
Output_linear = [
[-2, -1],
[ 1, 2]
]
```


Nếu nhân input ×2:

```lua
Z_new = [
[-4, -2],
[ 2, 4]
]

Output_linear_new = [
[-4, -2],
[ 2, 4]
]
```


➡️ Đầu vào thay đổi theo tỉ lệ bao nhiêu → đầu ra thay đổi đúng tỉ lệ đó.  
**→ Quan hệ tuyến tính.**

---

### 2. Có Activation (Phi tuyến)

#### a) ReLU

```lua
Output_relu = ReLU(Z) = [
[0, 0],
[1, 2]
]
```


Nếu nhân input ×2:

```lua
Output_relu_new = [
[0, 0],
[2, 4]
]
```


👉 Với số âm: -2 → -4 nhưng output vẫn 0.  
👉 Với số dương: giữ tỉ lệ (1→2, 2→4).  

➡️ Không đồng nhất → **phi tuyến**.

---

#### b) Tanh

```lua
Output_tanh = tanh(Z) ≈ [
[-0.96, -0.76],
[ 0.76, 0.96]
]
```


Nếu nhân input ×2:

```lua
Output_tanh_new ≈ [
[-0.999, -0.964],
[ 0.964, 0.999]
]
```


👉 Đầu ra không gấp đôi nữa, mà bị **nén lại** trong [-1, 1].

➡️ **Phi tuyến**.

---

## 🎯 Kết luận

- **Không Activation (tuyến tính):** đầu ra luôn thay đổi theo một tỉ lệ cố định.  
- **Có Activation (phi tuyến):** đầu ra thay đổi không theo tỉ lệ cố định, giúp mô hình học được quan hệ phức tạp hơn.
