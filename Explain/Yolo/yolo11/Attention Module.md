# Attention Module trong YOLO (Ultralytics)

## 1. Giới thiệu

Module **Attention** thực hiện cơ chế *multi-head self-attention* trên tensor đầu vào.  
Mục tiêu: cho phép mỗi vị trí (pixel) trong ảnh **nhìn thấy** và **tương tác** với mọi vị trí khác, giúp mô hình học được mối quan hệ không gian và ngữ cảnh.

---

## 2. Tham số quan trọng

- `dim`: số chiều kênh đầu vào (C).  
- `num_heads`: số lượng attention heads (mỗi head học 1 không gian con).  
- `attn_ratio`: tỉ lệ giữa kích thước key và head_dim.  

---

## 3. Các thuộc tính chính

- `head_dim = dim // num_heads`  
  → số chiều mỗi head sau khi chia.  

- `key_dim = int(head_dim * attn_ratio)`  
  → số chiều vector key (và query).  

- `scale = key_dim ** -0.5`  
  → hệ số chuẩn hóa khi tính attention (giúp ổn định gradient).  

- `qkv`: Conv1×1 sinh ra Q, K, V cùng lúc.  
- `proj`: Conv1×1 chiếu kết quả về lại `dim`.  
- `pe`: Conv3×3 (grouped by C) để thêm *positional encoding*.

---

## 4. Luồng xử lý (forward)

Giả sử input `x` có shape `(B, C, H, W)`:

1. **Bước 1 — Tính Q, K, V**

$$
\text{qkv} = \text{Conv}_{1\times1}(x)
$$

Sau đó reshape và split:

$$
[q, k, v] = \text{split}(\text{qkv}, [key\_dim, key\_dim, head\_dim])
$$

- q, k, v có shape `(B, num_heads, dim_sub, N)` với $N = H \times W$.

---

2. **Bước 2 — Attention score**

Tính similarity giữa query và key:

$$
\text{attn} = \frac{Q^T K}{\sqrt{d_k}}
$$

Sau đó chuẩn hóa bằng softmax theo mỗi hàng:

$$
\text{attn}_{ij} = \frac{\exp(s_{ij})}{\sum_j \exp(s_{ij})}
$$

---

3. **Bước 3 — Kết hợp với V**

Tạo đầu ra attention:

$$
O = V \cdot \text{attn}^T
$$

---

4. **Bước 4 — Positional Encoding (PE)**

Reshape lại O thành `(B, C, H, W)`, rồi cộng thêm positional encoding:

$$
O' = O + \text{PE}(O)
$$

---

5. **Bước 5 — Projection**

Chiếu tuyến tính về số kênh ban đầu:

$$
\text{Out} = \text{Conv}_{1\times1}(O')
$$

---

## 5. Ví dụ minh họa (ma trận 2 chiều)

Giả sử:

- `dim = 4`, `num_heads = 2`, `attn_ratio = 0.5`  
- Input: `(B=1, C=4, H=2, W=2)` → N = 4 tokens.  
- Chia 2 head: `head_dim = 2`, `key_dim = 1`.

Ma trận đầu vào (x):

| pixel | c0 | c1 | c2 | c3 |
|-------|----|----|----|----|
| p0    | 1  | 0  | 1  | 2  |
| p1    | 2  | 1  | 0  | 2  |
| p2    | 3  | 0  | 1  | 2  |
| p3    | 4  | 1  | 0  | 2  |

- Sau qkv: mỗi token được ánh xạ thành (q, k, v).  
- Sau softmax: mỗi token học **trọng số phân bổ** đến các token khác.  
- Sau nhân với V: mỗi token trở thành **tổ hợp tuyến tính** của toàn bộ tokens trong ảnh.  

---

## 6. Ý nghĩa

- Attention cho phép mô hình *không chỉ nhìn local (như conv)* mà còn *nhìn global*.  
- `num_heads` giúp học nhiều mối quan hệ song song.  
- `attn_ratio` điều chỉnh dung lượng key/query → cân bằng giữa tốc độ và độ chính xác.  
- `pe` bổ sung thông tin vị trí (Conv 3×3 group), giúp mô hình phân biệt pixel *trái/phải/trên/dưới*.  

> Đây là cách YOLOv8+ dùng **Attention** để cải thiện khả năng phát hiện vật thể trong ảnh phức tạp.

---
# Ví dụ chi tiết Attention — từng bước (dành cho người mới)

> Mục tiêu: cho bạn thấy **từ ma trận đầu vào (x)** qua các bước **tạo Q, K, V → tính score → softmax → weighted sum với V** ra được output như thế nào, từng phép toán một — không dùng code, chỉ tính tay trên ma trận 2-chiều.

---

## 1) Ma trận đầu vào (x)

| pixel | c0 | c1 | c2 | c3 |
|-------|----|----|----|----|
| p0    | 1  | 0  | 1  | 2  |
| p1    | 2  | 1  | 0  | 2  |
| p2    | 3  | 0  | 1  | 2  |
| p3    | 4  | 1  | 0  | 2  |

- N = 4 (số pixel / token)  
- C = 4 (số kênh)

> **Ghi chú:** đây là ví dụ *đơn giản hóa* để minh hoạ. Ở mô hình thật, Q/K/V được tính bằng phép nhân ma trận với trọng số học được, không phải "lấy thẳng" một cột như ví dụ này.

---

## 2) Giản lược để dễ hiểu (quy ước của ví dụ)

Để trình bày rõ ràng, ta giả sử:

- `Q` = cột `c0` (tham số đơn chiều cho mỗi pixel)  
- `K` = cột `c1`  
- `V` = hai thành phần `[c2, c3]` (vector 2 chiều)

Vậy ta có bảng Q, K, V:

| pixel | Q  | K  | V = [c2, c3] |
|-------|----|----|---------------|
| p0    | 1  | 0  | [1, 2]        |
| p1    | 2  | 1  | [0, 2]        |
| p2    | 3  | 0  | [1, 2]        |
| p3    | 4  | 1  | [0, 2]        |

> Lưu ý: trong ví dụ này mỗi Q, K là số vô hướng (scalar) — tức $d_k = 1$.

---

## 3) Công thức tính *score* (trước softmax)

> Giải thích: score thể hiện mức "tương đồng" giữa query của pixel i và key của pixel j. Nếu lớn → pixel i sẽ chú ý nhiều đến pixel j.

Công thức:

$$
\text{Score}(i,j) = \frac{Q_i \cdot K_j}{\sqrt{d_k}}
$$

- Ở ví dụ này $d_k = 1$ nên $\sqrt{d_k} = 1$ → không thay đổi giá trị.

---

## 4) Tính ma trận Score (Q×K)

Ta nhân Q_i với K_j theo tất cả cặp i,j:

|       | K(p0)=0 | K(p1)=1 | K(p2)=0 | K(p3)=1 |
|-------|---------|---------|---------|---------|
| Q(p0)=1 | 0       | 1       | 0       | 1       |
| Q(p1)=2 | 0       | 2       | 0       | 2       |
| Q(p2)=3 | 0       | 3       | 0       | 3       |
| Q(p3)=4 | 0       | 4       | 0       | 4       |

> Mỗi hàng i là vector score của pixel i với tất cả các pixel j.

---

## 5) Chuẩn hoá bằng Softmax (theo hàng)

> Giải thích: softmax biến score thành **trọng số α_{ij}** (không âm, tổng trên j bằng 1). Trọng số cho biết pixel i "trộn" bao nhiêu từ mỗi pixel j.

Công thức softmax (theo hàng):

$$
\alpha_{ij} = \frac{\exp(\text{Score}(i,j))}{\sum_{k=1}^{N} \exp(\text{Score}(i,k))}
$$

Ta cần tính $\exp(\text{score})$ cho từng phần tử. Dùng $e \approx 2.718281828$:

- $e^1 \approx 2.718281828$  
- $e^2 \approx 7.389056099$  
- $e^3 \approx 20.085536923$  
- $e^4 \approx 54.598150033$

Bây giờ tính từng hàng:

### Hàng i = p0 (scores = [0, 1, 0, 1])

- exp = [1, e, 1, e] = [1, 2.718281828, 1, 2.718281828]  
- tổng = $1 + e + 1 + e = 2 + 2e = 2 + 5.436563656 = 7.436563656$  
- trọng số:

  - α(p0→p0) = 1 / 7.436563656 ≈ **0.134366**  
  - α(p0→p1) = e / 7.436563656 ≈ **0.365634**  
  - α(p0→p2) = 1 / 7.436563656 ≈ **0.134366**  
  - α(p0→p3) = e / 7.436563656 ≈ **0.365634**

> Kiểm tra: tổng ≈ 0.134366 + 0.365634 + 0.134366 + 0.365634 = 1.000000

### Hàng i = p1 (scores = [0, 2, 0, 2])

- exp = [1, e^2, 1, e^2] = [1, 7.389056099, 1, 7.389056099]  
- tổng = 2 + 2·7.389056099 = 16.778112198  
- trọng số:

  - α(p1→p0) = 1 / 16.778112198 ≈ **0.059612**  
  - α(p1→p1) = 7.389056099 / 16.778112198 ≈ **0.440388**  
  - α(p1→p2) = 1 / 16.778112198 ≈ **0.059612**  
  - α(p1→p3) = 7.389056099 / 16.778112198 ≈ **0.440388**

### Hàng i = p2 (scores = [0, 3, 0, 3])

- exp = [1, e^3, 1, e^3] = [1, 20.085536923, 1, 20.085536923]  
- tổng = 2 + 2·20.085536923 = 42.171073846  
- trọng số:

  - α(p2→p0) = 1 / 42.171073846 ≈ **0.023712**  
  - α(p2→p1) = 20.085536923 / 42.171073846 ≈ **0.476288**  
  - α(p2→p2) = 1 / 42.171073846 ≈ **0.023712**  
  - α(p2→p3) = 20.085536923 / 42.171073846 ≈ **0.476288**

### Hàng i = p3 (scores = [0, 4, 0, 4])

- exp = [1, e^4, 1, e^4] = [1, 54.598150033, 1, 54.598150033]  
- tổng = 2 + 2·54.598150033 = 111.196300066  
- trọng số:

  - α(p3→p0) = 1 / 111.196300066 ≈ **0.008989**  
  - α(p3→p1) = 54.598150033 / 111.196300066 ≈ **0.491011**  
  - α(p3→p2) = 1 / 111.196300066 ≈ **0.008989**  
  - α(p3→p3) = 54.598150033 / 111.196300066 ≈ **0.491011**

> Tổng kết ma trận α (làm tròn 6 chữ số):

|       | p0       | p1       | p2       | p3       |
|-------|----------|----------|----------|----------|
| p0    | 0.134366 | 0.365634 | 0.134366 | 0.365634 |
| p1    | 0.059612 | 0.440388 | 0.059612 | 0.440388 |
| p2    | 0.023712 | 0.476288 | 0.023712 | 0.476288 |
| p3    | 0.008989 | 0.491011 | 0.008989 | 0.491011 |

---

## 6) Tính output: trộn V theo trọng số α

Công thức:

$$
\text{Out}(i) = \sum_{j=1}^{N} \alpha_{ij} \cdot V_j
$$

V_j = [c2_j, c3_j], với c2_j ∈ {0,1} và c3_j = 2 cho mọi j.

Danh sách V:

- V(p0) = [1, 2]  
- V(p1) = [0, 2]  
- V(p2) = [1, 2]  
- V(p3) = [0, 2]

Tính từng pixel:

### Pixel p0

- out_c2 = 0.134366·1 + 0.365634·0 + 0.134366·1 + 0.365634·0  
  = 0.134366 + 0 + 0.134366 + 0 = **0.268732**

- out_c3 = 0.134366·2 + 0.365634·2 + 0.134366·2 + 0.365634·2  
  = 2 · (0.134366 + 0.365634 + 0.134366 + 0.365634) = 2 · 1 = **2.000000**

→ Out(p0) ≈ **[0.268732, 2.000000]**

### Pixel p1

- out_c2 = 0.059612·1 + 0.440388·0 + 0.059612·1 + 0.440388·0  
  = 0.059612 + 0 + 0.059612 + 0 = **0.119224**

- out_c3 = 2 · 1 = **2.000000**

→ Out(p1) ≈ **[0.119224, 2.000000]**

### Pixel p2

- out_c2 = 0.023712·1 + 0.476288·0 + 0.023712·1 + 0.476288·0  
  = 0.023712 + 0 + 0.023712 + 0 = **0.047424**

- out_c3 = 2 · 1 = **2.000000**

→ Out(p2) ≈ **[0.047424, 2.000000]**

### Pixel p3

- out_c2 = 0.008989·1 + 0.491011·0 + 0.008989·1 + 0.491011·0  
  = 0.008989 + 0 + 0.008989 + 0 = **0.017978**

- out_c3 = 2 · 1 = **2.000000**

→ Out(p3) ≈ **[0.017978, 2.000000]**

---

## 7) Bảng kết quả (output)

| pixel | out_c2    | out_c3  |
|-------|-----------|---------|
| p0    | 0.268732  | 2.000000|
| p1    | 0.119224  | 2.000000|
| p2    | 0.047424  | 2.000000|
| p3    | 0.017978  | 2.000000|

---

## 8) Giải thích trực quan — vì sao kết quả như vậy?

- **Tại sao K(p1) = 1 và K(p0) = 0?**  
  Vì trong ví dụ đơn giản ta *giả sử* K = c1. Do c1 tại p1 và p3 bằng 1, còn p0/p2 bằng 0 → K(p1)=K(p3)=1, K(p0)=K(p2)=0.

- **Tại sao out_c3 luôn = 2?**  
  Vì ở mọi pixel j, V_j.c3 = 2 (hằng số). Weighted sum của các 2 sẽ vẫn bằng 2 (vì tổng α trên mỗi hàng = 1). Vì vậy attention không thay đổi thành phần c3 trong ví dụ này.

- **Tại sao out_c2 giảm khi Q tăng (p0→p3)?**  
  - Khi Q_i lớn hơn, score(i, j) = Q_i * K_j lớn hơn cho các j có K_j = 1.  
  - Softmax phân bổ nhiều trọng số cho các j có K=1 (tức p1 và p3). Trong ví dụ, p1/p3 có V_c2 = 0.  
  - Do đó pixel i “chú ý” nhiều đến token có c2 = 0, làm cho out_c2 (trung bình có trọng số) giảm.  
  - Kết quả: p3 (Q=4) có out_c2 nhỏ nhất ~0.01798; p0 (Q=1) ít chú ý tới K=1 hơn nên out_c2 cao hơn ~0.2687.

> Đây là minh chứng trực quan cho ý tưởng: **attention dùng mối tương đồng Q·K để chọn thông tin từ V**. Nếu những vị trí mà i chú ý nhiều có c2=0 thì out_c2 sẽ thấp; nếu chú ý nhiều tới vị trí có c2=1 thì out_c2 sẽ cao.

---

## 9) Kết luận & liên hệ thực tế

- Ví dụ này **rất giản lược** (Q,K,V lấy trực tiếp cột), nhưng cho thấy chính xác cách *score → softmax → weighted sum* vận hành và ảnh hưởng tới từng thành phần của V.  
- Trong mô hình thật:
  - Q, K, V được tạo bằng phép nhân với ma trận trọng số $W_Q, W_K, W_V$ (học được).  
  - K và V thường không có giá trị hằng như ví dụ; do đó output có thể thay đổi cả về cấu trúc và ý nghĩa.  
  - Thêm `scale = 1/\sqrt{d_k}` để ổn định giá trị score khi $d_k$ lớn.  
  - Multi-head attention chia không gian đặc trưng thành nhiều head để học nhiều kiểu tương đồng khác nhau.

> Nếu bạn muốn, mình sẽ **làm lại ví dụ này nhưng dùng W_Q, W_K, W_V cụ thể (ma trận nhỏ 4×2)** để thấy Q,K,V sinh ra từ phép nhân ma trận như thế nào — rồi lặp lại toàn bộ phép tính trên. Bạn muốn mình làm tiếp theo hướng đó chứ?
