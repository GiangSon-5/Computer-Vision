# 📍 Vị trí không gian trong Object Detection (YOLO)

Hình minh họa cho thấy cách mô hình object detection (như **YOLO**) sử dụng thông tin **vị trí không gian** để phát hiện đối tượng.

---

## 1. Grid trên ảnh
- Ảnh con chó được chia thành một **lưới (grid)**, ví dụ 13×13 hoặc 19×19 (tùy kích thước ảnh và kiến trúc).  
- Mỗi **ô lưới** đại diện cho một vùng không gian cụ thể trong ảnh gốc.  
- Mỗi ô có thể dự đoán **một hoặc nhiều bounding box** nếu phát hiện có đối tượng trong vùng đó.  

> Đây là cách mô hình **giữ lại thông tin không gian**: mỗi ô biết mình đang “nhìn” vào vùng nào của ảnh.

---

## 2. Bounding Box (Hộp giới hạn)
- **Hộp vàng**: bounding box chính xác mà mô hình dự đoán, bao quanh toàn bộ con chó.  
- **Hộp đỏ**: có thể là một dự đoán chưa chính xác hoặc một **anchor box** trong quá trình huấn luyện.  

Bounding box được biểu diễn bằng **4 tham số**:

- $$t_x, t_y$$ : vị trí tâm hộp (tương đối với ô lưới)  
- $$t_w, t_h$$ : chiều rộng và chiều cao (tương đối với ảnh hoặc anchor box)

---

## 3. Prediction Feature Map
Là **tensor đầu ra** của mô hình sau khi xử lý ảnh qua *Backbone + Neck*.

- Mỗi **cell trong feature map** tương ứng với một ô lưới trong ảnh gốc.  
- Mỗi cell sẽ dự đoán:
  - **Box Coordinates**: $$t_x, t_y, t_w, t_h$$  
  - **Objectness Score**: $$p_o$$ → độ tin cậy có đối tượng  
  - **Class Scores**: $$p_1, p_2, ..., p_c$$ → xác suất thuộc từng lớp  

Tất cả các thông tin này đều **gắn liền với vị trí không gian** của ô lưới tương ứng.

---

## 4. Tính toán vị trí thực tế
Tọa độ thực tế của bounding box được tính từ các giá trị dự đoán như sau:

$$
b_x = \sigma(t_x) + c_x
$$  

$$
b_y = \sigma(t_y) + c_y
$$  

$$
b_w = p_w \cdot e^{t_w}
$$  

$$
b_h = p_h \cdot e^{t_h}
$$  

Trong đó:
- $$(c_x, c_y)$$ : tọa độ gốc của ô lưới  
- $$\sigma$$ : hàm sigmoid để chuẩn hóa giá trị  
- $$(p_w, p_h)$$ : kích thước anchor box  

---

## ✅ Tổng kết: Vị trí không gian là gì?

| Thành phần | Vai trò không gian |
|------------|--------------------|
| Grid trên ảnh | Chia ảnh thành các vùng có vị trí rõ ràng |
| Feature Map | Mỗi cell tương ứng với một vùng trong ảnh |
| Bounding Box | Dự đoán vị trí và kích thước đối tượng |
| Tọa độ $$(t_x, t_y, t_w, t_h)$$ | Mã hóa vị trí tương đối trong ảnh |
