# Backbone trong YOLOv11

[VÍ DỤ BACKBONE](../yolo11/EX/Backbone_yolo11_Ex.md)

Backbone là **thành phần đầu tiên và quan trọng nhất** trong YOLOv11. Nó đóng vai trò như *bộ trích xuất đặc trưng* (feature extractor), biến ảnh đầu vào thành *các bản đồ đặc trưng (feature maps)* giàu thông tin để chuyển tiếp cho **Neck** và **Head**.

---

## 1. Backbone trong bối cảnh Ba phần chính

YOLOv11 được chia thành 3 phần chính:

- **Backbone**: Trích xuất đặc trưng từ ảnh đầu vào.  
- **Neck**: Kết hợp đặc trưng đa mức, tăng tính biểu đạt.  
- **Head**: Sinh ra dự đoán (bounding box + class).

Đặc điểm Backbone:

- Bắt đầu bằng **2 lớp convolution** (`kernel=3, stride=2`) → giảm độ phân giải ảnh xuống **một nửa**.  
- Xen kẽ nhiều khối **C3 K2** (số lượng phụ thuộc *depth multiple*).  
- Số kênh đầu ra tính theo công thức:

$$
\text{OutputChannels} = \min(\text{BaseChannel}, \text{MaxChannel}) \times \text{WidthMultiple}
$$

⚡ **[VÍ DỤ OutputChannels](../../Yolo/yolo11/EX/OutputChannels_EX.md)**


- Kết quả được chuyển cho **Neck** (ví dụ SPF).  
- Các phiên bản YOLOv11 (`n, s, m, l, xl`) khác nhau ở: `depth multiple, width multiple, max channels`.

⚡ **[VÍ DỤ các phiên bản YOLO11](../yolo11/yolo11_n_S_m_l_xl.md)**

---

## 2. Trình trích xuất đặc trưng trong bối cảnh Backbone

Backbone chính là **trình trích xuất đặc trưng**:

- Học ra các đặc trưng như cạnh, góc, họa tiết, vùng quan trọng trong ảnh.  
- Được xây dựng từ chuỗi **convolution + khối C3 K2**.  
- Một số đầu ra trung gian của C3 K2 được truyền thẳng sang **Neck** để hỗ trợ tổng hợp đa cấp.

> *Backbone là “con mắt” của YOLOv11. Nếu thiếu Backbone, Neck và Head sẽ không có thông tin để dự đoán đối tượng.*

---

## 3. Nhiều lớp Convolution trong bối cảnh Backbone

Các lớp convolution là nền tảng của Backbone:

- Lớp conv đầu tiên: `kernel=3, stride=2` → giảm ảnh từ $640 \times 640$ thành $320 \times 320$.  
- Sau đó: xen kẽ **conv stride=2** và **khối C3 K2**.  
- Mỗi lớp conv giúp học đặc trưng ở mức độ trừu tượng ngày càng cao.

> *Không chỉ giảm kích thước ảnh, convolution còn trực tiếp học ra biểu diễn đặc trưng từ pixel.*

---

## 4. Giảm độ phân giải trong bối cảnh Backbone

Quá trình giảm độ phân giải diễn ra bằng **convolution với stride=2**:

- 2 lớp conv đầu tiên đã giảm ảnh từ $640 \times 640$ → $320 \times 320$.  
- Các conv stride=2 sau đó tiếp tục giảm kích thước.  
- Mục đích:  
  - Trích xuất đặc trưng ở nhiều mức độ (low-level, high-level).  
  - Mở rộng vùng cảm thụ (receptive field).  
  - Giảm chi phí tính toán.

> *Giảm độ phân giải là cơ chế quan trọng để CNN vừa nắm bắt ngữ cảnh rộng vừa giữ lại đặc trưng cốt lõi.*

---

# ✅ Tóm tắt

- **Backbone = Feature extractor** của YOLOv11.  
- Được xây dựng từ **chuỗi convolution + khối C3 K2**.  
- Dùng **stride=2** để giảm độ phân giải dần dần.  
- Đầu ra phụ thuộc vào `depth multiple, width multiple, max channel`.  
- Cung cấp **feature maps đa cấp** cho Neck và Head để thực hiện detection.


