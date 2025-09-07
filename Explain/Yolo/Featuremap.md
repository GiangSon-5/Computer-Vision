## 📐 Các kích thước Feature Map: 13×13, 26×26, 52×52

> 🔗 **[Giải thích head - stride](../Yolo/stride_32.md)**

![3_Boudingbox](../../imgs/3_boudingbox.jpg)

Khi ảnh đầu vào (ví dụ **416×416**) đi qua mạng CNN, nó được giảm kích thước dần qua các lớp convolution và pooling.  
Kết quả là ta thu được **các feature map** có kích thước nhỏ hơn:

| Feature Map | Dự đoán cho    | Đặc điểm |
|-------------|----------------|----------|
| 13×13       | Vật thể lớn    | Nhìn tổng thể, ít chi tiết |
| 26×26       | Vật thể vừa    | Cân bằng giữa chi tiết và tổng thể |
| 52×52       | Vật thể nhỏ    | Nhìn chi tiết, độ phân giải cao |

---

### 🎯 Tại sao cần nhiều feature map?
- Vật thể nhỏ (ví dụ: cái ly, con mèo con) → rất khó phát hiện nếu chỉ dùng feature map 13×13.  
- Vật thể lớn (ví dụ: ô tô, con người) → không cần độ chi tiết quá cao.  

👉 Vì vậy, **YOLOv3 và các phiên bản sau** dùng **3 feature map song song** để:
- Dự đoán vật thể ở nhiều kích thước khác nhau.  
- Tăng độ chính xác tổng thể.  

---

### 📦 Mỗi feature map có bao nhiêu ô dự đoán?
Với mỗi feature map, **mỗi ô (cell)** sẽ dự đoán **3 anchor boxes**.  

Tổng số dự đoán trên một ảnh là:

$$
(13 \times 13 + 26 \times 26 + 52 \times 52) \times 3 = 10647 \ \text{bounding boxes}
$$

→ Mỗi ảnh sẽ sinh ra **hơn 10,000 hộp dự đoán**! Sau đó, thuật toán **Non-Max Suppression (NMS)** được dùng để lọc ra các hộp tốt nhất.

---

## 🧠 Tổng kết

| Kích thước | Dự đoán cho   | Số ô (cells) | Số anchor box |
|------------|---------------|--------------|---------------|
| 13×13      | Vật thể lớn   | 169          | 507           |
| 26×26      | Vật thể vừa   | 676          | 2028          |
| 52×52      | Vật thể nhỏ   | 2704         | 8112          |
| **Tổng**   | —             | 3549         | **10647**     |

➡️ Nhờ kết hợp **nhiều tỉ lệ feature map** và **anchor boxes**, YOLO có khả năng phát hiện **đa dạng kích thước vật thể** trong cùng một bức ảnh.
