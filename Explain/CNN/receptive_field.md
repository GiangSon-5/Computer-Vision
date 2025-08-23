# Receptive Field trong Computer Vision

## 1. Giới thiệu
Trong *computer vision*, khái niệm **receptive field (RF)** rất quan trọng để hiểu cách mô hình học được đặc trưng từ ảnh.  
- *Receptive field* mô tả vùng ảnh gốc (input image) mà một neuron ở một tầng (layer) nào đó "nhìn thấy" hoặc chịu ảnh hưởng.  
- Khi đi sâu hơn vào mạng, receptive field càng lớn, tức là neuron ở tầng cao sẽ tổng hợp thông tin từ vùng ảnh rộng hơn.  

> **Ví dụ:** Ở tầng đầu tiên, một filter 3×3 chỉ nhìn được 9 pixel gốc. Nhưng ở tầng thứ 3, cùng một neuron có thể đã tổng hợp thông tin từ hàng chục pixel ở ảnh gốc.

---

## 2. Tại sao Receptive Field quan trọng?
- **Phân giải chi tiết (local vs global):**  
  - RF nhỏ → học đặc trưng chi tiết (cạnh, góc, kết cấu).  
  - RF lớn → học ngữ cảnh tổng thể (vật thể, bố cục).  

- **Thiết kế mô hình:**  
  Hiểu RF giúp chọn kích thước kernel, stride, padding phù hợp để đảm bảo mô hình "nhìn" đủ rộng.  

- **Ứng dụng:**  
  - Object detection cần RF lớn để bao quát vật thể.  
  - Semantic segmentation cần RF phù hợp để cân bằng giữa chi tiết và ngữ cảnh.

---

## 3. Công thức tính Receptive Field
Khi thiết kế CNN, receptive field tại tầng `l` được tính dựa vào:
- Kernel size ($k_l$)  
- Stride ($s_l$)  
- Padding ($p_l$)  
- Receptive field và *jump* (tỉ lệ phóng đại bước nhảy trên ảnh gốc) từ tầng trước.

### 3.1. Kích thước receptive field
Giả sử $RF_l$ là receptive field tại tầng $l$, công thức tổng quát:

$$
RF_l = RF_{l-1} + (k_l - 1) \times J_{l-1}
$$

Trong đó:
- $RF_{l-1}$: receptive field tại tầng trước.  
- $k_l$: kernel size tại tầng $l$.  
- $J_{l-1}$: *jump* từ tầng trước (xem mục dưới).  

---

### 3.2. Công thức tính *jump*
*Jump* ($J_l$) mô tả "bước nhảy" trong ảnh gốc khi dịch 1 pixel ở tầng $l$.

$$
J_l = J_{l-1} \times s_l
$$

Trong đó:
- $s_l$: stride của tầng $l$.  
- $J_{l-1}$: jump tại tầng trước.  

> Ghi chú: Ban đầu, ở tầng input (ảnh gốc), ta có $RF_0 = 1$ và $J_0 = 1$.

---
---

# Receptive Field trong Computer Vision

## 1. Giới thiệu
Trong *computer vision*, khái niệm **receptive field (RF)** rất quan trọng để hiểu cách mô hình học được đặc trưng từ ảnh.  
- *Receptive field* mô tả vùng ảnh gốc (input image) mà một neuron ở một tầng (layer) nào đó "nhìn thấy" hoặc chịu ảnh hưởng.  
- Khi đi sâu hơn vào mạng, receptive field càng lớn, tức là neuron ở tầng cao sẽ tổng hợp thông tin từ vùng ảnh rộng hơn.  

> **Ví dụ:** Ở tầng đầu tiên, một filter 3×3 chỉ nhìn được 9 pixel gốc. Nhưng ở tầng thứ 3, cùng một neuron có thể đã tổng hợp thông tin từ hàng chục pixel ở ảnh gốc.

---

## 2. Tại sao Receptive Field quan trọng?
- **Phân giải chi tiết (local vs global):**  
  - RF nhỏ → học đặc trưng chi tiết (cạnh, góc, kết cấu).  
  - RF lớn → học ngữ cảnh tổng thể (vật thể, bố cục).  

- **Thiết kế mô hình:**  
  Hiểu RF giúp chọn kích thước kernel, stride, padding phù hợp để đảm bảo mô hình "nhìn" đủ rộng.  

- **Ứng dụng:**  
  - Object detection cần RF lớn để bao quát vật thể.  
  - Semantic segmentation cần RF phù hợp để cân bằng giữa chi tiết và ngữ cảnh.

---

## 3. Công thức tính Receptive Field
Khi thiết kế CNN, receptive field tại tầng `l` được tính dựa vào:
- Kernel size ($k_l$)  
- Stride ($s_l$)  
- Padding ($p_l$)  
- Receptive field và *jump* (tỉ lệ phóng đại bước nhảy trên ảnh gốc) từ tầng trước.

### 3.1. Kích thước receptive field
Giả sử $RF_l$ là receptive field tại tầng $l$, công thức tổng quát:

$$
RF_l = RF_{l-1} + (k_l - 1) \times J_{l-1}
$$

Trong đó:
- $RF_{l-1}$: receptive field tại tầng trước.  
- $k_l$: kernel size tại tầng $l$.  
- $J_{l-1}$: *jump* từ tầng trước (xem mục dưới).  

---

### 3.2. Công thức tính *jump*
*Jump* ($J_l$) mô tả "bước nhảy" trong ảnh gốc khi dịch 1 pixel ở tầng $l$.

$$
J_l = J_{l-1} \times s_l
$$

Trong đó:
- $s_l$: stride của tầng $l$.  
- $J_{l-1}$: jump tại tầng trước.  

> Ghi chú: Ban đầu, ở tầng input (ảnh gốc), ta có $RF_0 = 1$ và $J_0 = 1$.

---

## 4. Ví dụ minh họa
Giả sử ta có một CNN với 3 tầng convolution (không padding):

- Tầng 1: kernel = 3, stride = 1  
- Tầng 2: kernel = 3, stride = 2  
- Tầng 3: kernel = 3, stride = 1  

### Bước tính:
- Tầng 1:  
  $RF_1 = 1 + (3-1)\times1 = 3$, $J_1 = 1\times1 = 1$  
- Tầng 2:  
  $RF_2 = 3 + (3-1)\times1 = 5$, $J_2 = 1\times2 = 2$  
- Tầng 3:  
  $RF_3 = 5 + (3-1)\times2 = 9$, $J_3 = 2\times1 = 2$  

Kết quả: một neuron ở tầng 3 nhìn thấy **9 pixel** từ ảnh gốc.

---

## 5. Kết luận
- **Receptive field** là công cụ giúp hiểu "tầm nhìn" của mô hình.  
- **Công thức** dựa vào kernel, stride, padding và jump.  
- **Ứng dụng thực tế:** tối ưu kiến trúc CNN cho detection, segmentation, recognition.  

> Hãy nhớ: càng sâu, receptive field càng rộng, nhưng nếu thiết kế không khéo, mô hình có thể "không nhìn thấy" toàn bộ ngữ cảnh mong muốn.


# Liên hệ Công Thức với Ví Dụ Receptive Field

---

## 1. Công thức nhắc lại

- Receptive field tại tầng `l`:

$$
RF_l = RF_{l-1} + (k_l - 1) \times J_{l-1}
$$

- Jump tại tầng `l`:

$$
J_l = J_{l-1} \times s_l
$$

Trong đó:
- $k_l$: kernel size  
- $s_l$: stride  
- $J_l$: khoảng "bước nhảy" quy chiếu về ảnh gốc  
- Ban đầu: $RF_0 = 1$, $J_0 = 1$

---

## 2. Ví dụ cụ thể (3 tầng CNN)

Giả sử có 3 tầng convolution, **kernel = 3, stride = 1, không padding**.

### Tầng 1
- $RF_1 = RF_0 + (k_1 - 1) \times J_0 = 1 + (3-1)\times1 = 3$  
- $J_1 = J_0 \times s_1 = 1\times1 = 1$  

➡️ Neuron tầng 1 nhìn được **3 pixel gốc**.

---

### Tầng 2
- $RF_2 = RF_1 + (k_2 - 1) \times J_1 = 3 + (3-1)\times1 = 5$  
- $J_2 = J_1 \times s_2 = 1\times1 = 1$  

➡️ Neuron tầng 2 nhìn được **5 pixel gốc**.

---

### Tầng 3
- $RF_3 = RF_2 + (k_3 - 1) \times J_2 = 5 + (3-1)\times1 = 7$  
- $J_3 = J_2 \times s_3 = 1\times1 = 1$  

➡️ Neuron tầng 3 nhìn được **7 pixel gốc**.

---

## 3. Ý nghĩa Jump trong ví dụ này
Vì stride = 1 ở mọi tầng ⇒ $J$ luôn = 1.  
- Điều này có nghĩa: **dịch 1 pixel ở tầng cao** thì cũng dịch đúng **1 pixel trên ảnh gốc**.  

Nếu stride > 1 thì $J$ sẽ tăng nhanh, và receptive field sẽ mở rộng **nhanh hơn**.

---

## 4. So sánh để dễ nhớ
- Nếu chỉ **nhớ trực giác**: mỗi tầng kernel 3×3 tăng thêm 2 pixel cho RF.  
- Nếu dùng **công thức**: ta sẽ tính được cả trường hợp stride > 1 hoặc padding khác.  

> Ví dụ: nếu tầng 2 có stride = 2, thì $J_2 = 2$, và RF sẽ nhảy tăng mạnh hơn.

---

## 5. Kết luận
- Công thức RF + Jump giúp ta **tính chính xác receptive field** ở bất kỳ tầng nào.  
- Trong ví dụ stride = 1, kernel = 3:  
  - $RF = 3, 5, 7$  
  - $J = 1$ ở mọi tầng  
- Nếu stride lớn hơn, $J$ sẽ đóng vai trò **phóng đại bước nhảy**, làm RF mở rộng nhanh hơn.


---
---

## . Kết luận
- **Receptive field** là công cụ giúp hiểu "tầm nhìn" của mô hình.  
- **Công thức** dựa vào kernel, stride, padding và jump.  
- **Ứng dụng thực tế:** tối ưu kiến trúc CNN cho detection, segmentation, recognition.  

> Hãy nhớ: càng sâu, receptive field càng rộng, nhưng nếu thiết kế không khéo, mô hình có thể "không nhìn thấy" toàn bộ ngữ cảnh mong muốn.
