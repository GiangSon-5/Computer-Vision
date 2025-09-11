

# 🔗 Skip Connection (kết nối tắt)

## 1. Ý nghĩa

* Skip connection = **nối tắt từ input sang output**, thay vì chỉ qua nhiều lớp liên tiếp.
* Công thức:

$$
y = f(x) + x
$$

Trong đó:

* $x$: input gốc (đặc trưng ban đầu).
* $f(x)$: output sau khi qua khối xử lý (Conv, Attention,…).
* $y$: kết quả cuối cùng sau khi cộng.

---

## 2. Lợi ích chính

### 🚀 Giữ lại thông tin gốc

Nếu đi qua nhiều lớp, đặc trưng có thể bị méo/mất.
Skip connection giữ lại input để “bơm thẳng” vào output → mô hình không quên thông tin ban đầu.

---

### 🚀 Giảm gradient vanish

Trong mạng sâu, gradient dễ bị tiêu biến.
Skip connection mở một “đường tắt” cho gradient quay ngược lại, giúp việc học ổn định hơn.

---

### 🚀 Học phần dư (Residual Learning)

Mạng không học toàn bộ ánh xạ $H(x)$ nữa, mà chỉ học phần dư:

$$
H(x) = f(x) + x 
\quad \Rightarrow \quad
f(x) = H(x) - x
$$

→ Dễ học hơn nhiều, vì chỉ cần “sửa lỗi” thay vì xây mới hoàn toàn.

---

### 🚀 Kết hợp nhiều mức đặc trưng

* $x$: đặc trưng thô (cạnh, màu).
* $f(x)$: đặc trưng cao (ngữ cảnh, cấu trúc).
* Khi cộng lại: vừa chi tiết, vừa ngữ cảnh → dự đoán chính xác hơn.

---

## 3. Ví dụ toán học

### 🔴 Trường hợp không có skip connection

Giả sử muốn mạng học ánh xạ:

$$
H(x) = 2x
$$

Nếu bắt mạng học trực tiếp $H(x)$, nó phải tìm đúng quy luật gấp đôi.

---

### 🟢 Trường hợp có skip connection

Ta viết lại:

$$
H(x) = f(x) + x
$$

Khi đó:

$$
f(x) = H(x) - x = 2x - x = x
$$

👉 Mạng chỉ cần học $f(x) = x$ (bản sao đơn giản), thay vì $H(x) = 2x$ (gấp đôi).
Dễ dàng hơn rất nhiều!

---

## 4. Trực quan (ví dụ hình ảnh)

* Input $x$: ảnh hơi mờ → đã chứa 80% thông tin.
* Output mong muốn $H(x)$: ảnh rõ.
* Phần dư $f(x)$: chi tiết còn thiếu (20% độ nét).

Khi có skip connection, mạng **chỉ cần học phần chi tiết còn thiếu** thay vì phải dựng lại toàn bộ ảnh từ đầu.

---

## 5. Tóm gọn

Skip connection giúp:

1. Giữ thông tin gốc.
2. Tránh gradient vanish.
3. Học phần dư → nhanh & ổn định hơn.
4. Kết hợp đặc trưng nhiều mức → tăng chính xác.

👉 **Phần dư $f(x)$ chính là những gì cần thêm/sửa để biến input thành output mong muốn.**

---




---

## 1. Đang có hai “ngữ cảnh”

### Ngữ cảnh A — **Code chạy (forward)**

Mạng chạy thật thì nó chỉ biết tính toán:

$$
y = f(x) + x
$$

* $x$: input của block.
* $f(x)$: một chuỗi conv/bn/relu bên trong block.
* $y$: output mà block xuất ra.

---

### Ngữ cảnh B — **Lý thuyết học (target function)**

Khi ta thiết kế mạng, ta muốn mạng mô phỏng **một hàm mong muốn** $H(x)$.

Với residual learning, ta giả định:

$$
H(x) = f(x) + x
$$

* $H(x)$: mapping thật mà ta muốn học (ground-truth).
* $f(x)$: phần còn thiếu (residual) mà mạng cần học.

---

## 2. Mối quan hệ

* Trong **code**: ta luôn tính `y = f(x) + x`.
* Trong **phân tích**: ta so sánh $y$ với $H(x)$.

Nếu $y$ khớp với $H(x)$, tức là mạng đã học được residual $f(x) = H(x) - x$.

---

## 3. Ví dụ minh họa 

Giả sử ta muốn học $H(x) = 2x$.

* **Không có skip connection:**
  Mạng phải học trực tiếp $H(x) = 2x$.
  → khó hơn.

* **Có skip connection:**
  Block xuất ra $y = f(x) + x$.
  Muốn $y = 2x$, thì:

  $$
  f(x) = H(x) - x = 2x - x = x
  $$

  → mạng chỉ cần học “copy input” (đơn giản hơn nhiều).

---

## 4. Sơ đồ ASCII (để dễ hình dung)

```
Input x -----> [ f(x) ] ---+
             (Conv/BN/ReLU) |
                            +---> y  (output)
             Skip ----------+
```

* Forward: `y = f(x) + x`.
* Target: ta muốn `y ≈ H(x)`.
* Do đó, f(x) chỉ cần học phần **chênh lệch** giữa $H(x)$ và $x$.

---

👉 Nói ngắn gọn:

* `y = f(x) + x` là **cách tính toán trong mạng**.
* `H(x) = f(x) + x` là **cách chúng ta diễn giải bài toán học residual**.



---

## So sánh học hàm $H(x) = 2x$

| Trường hợp          | Công thức output | Mạng phải học gì?                                         | Độ khó                                           |
| ------------------- | ---------------- | --------------------------------------------------------- | ------------------------------------------------ |
| ❌ **Không có skip** | $y = f(x)$     | $f(x) = H(x) = 2x$ (phải học toàn bộ phép nhân đôi)     | Khó hơn (mạng phải tái tạo cả hàm $2x$ từ đầu) |
| ✅ **Có skip**       | $y = f(x) + x$ | Muốn $y = H(x) = 2x$ ⇒ $f(x) = H(x) - x = 2x - x = x$ | Dễ hơn (chỉ cần học “copy input”)                |

---

### Diễn giải

* **Không skip:** mô hình phải tìm ra cách biến $x$ thành $2x$ → khá tốn công.
* **Có skip:** mô hình đã có sẵn $+x$ từ đường tắt, nên chỉ cần học thêm phần dư $f(x) = x$ → dễ và nhanh hội tụ hơn.

---

👉 Vậy **skip connection làm bài toán trở nên “dễ” hơn**, vì mạng chỉ học **phần thiếu** thay vì toàn bộ hàm.

