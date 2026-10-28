# CoTMix kết hợp DINOv3 trên bộ dữ liệu UCI-HAR

## 1. Giới thiệu

Tài liệu này trình bày quy trình tích hợp bộ mã hóa dựa trên DINOv3 vào quy trình huấn luyện CoTMix cho bài toán nhận dạng hoạt động của con người trên bộ dữ liệu UCI-HAR. Mục tiêu của dự án là đọc dữ liệu cảm biến quán tính, chuyển dữ liệu sang định dạng mà CoTMix có thể sử dụng, trích xuất đặc trưng bằng bộ mã hóa được tích hợp trong mô hình, sau đó sử dụng các đặc trưng này trong quá trình huấn luyện và đánh giá.

Phần mô tả về bộ mã hóa tập trung vào mục đích tích hợp và các bước cần kiểm tra trong quy trình xử lý. Những thay đổi cụ thể về kiến trúc chỉ có thể được khẳng định sau khi đối chiếu mã nguồn của bộ mã hóa hiện tại với phiên bản gốc. Vì vậy, các nội dung chưa được xác minh trực tiếp từ mã nguồn được trình bày như những điểm cần kiểm tra, không phải kết luận về việc triển khai.

## 2. Bộ mã hóa và cơ sở thiết kế

### 2.1. Vai trò của bộ mã hóa

Bộ mã hóa có nhiệm vụ chuyển dữ liệu đầu vào thành biểu diễn đặc trưng để các thành phần phía sau sử dụng trong quá trình học và phân loại hoạt động. Trong bộ dữ liệu UCI-HAR, mỗi cửa sổ dữ liệu gồm 9 kênh cảm biến, mỗi kênh chứa 128 bước thời gian. Do đó, tensor đầu vào có dạng `[batch_size, 9, 128]`, trong đó `batch_size` là số mẫu được xử lý trong một lượt.

Dữ liệu UCI-HAR là chuỗi thời gian nhiều kênh, trong khi DINOv3 được phát triển theo hướng mô hình thị giác. Vì vậy, khi tích hợp bộ mã hóa dựa trên DINOv3, cần bảo đảm dữ liệu cảm biến được chuyển sang dạng đầu vào phù hợp với kiến trúc được sử dụng. Không thể mặc định rằng một bộ mã hóa được thiết kế cho ảnh có thể tiếp nhận trực tiếp tensor cảm biến mà không cần điều chỉnh.

### 2.2. Các thành phần cần đối chiếu

Việc xác định chính xác những thay đổi trong bộ mã hóa cần dựa trên so sánh mã nguồn gốc và mã nguồn hiện tại. Trước hết, cần kiểm tra kích thước và kiểu dữ liệu đầu vào để xác nhận dữ liệu HAR được tiếp nhận đúng cách. Tiếp theo, cần làm rõ cách lớp đầu vào hoặc phép chiếu chuyển 9 kênh cảm biến thành biểu diễn mà bộ mã hóa có thể xử lý. Đối với phần mạng chính, cần xác định những khối xử lý và trọng số nào được kế thừa từ DINOv3.

Ngoài ra, cần kiểm tra kích thước và số chiều của biểu diễn đặc trưng đầu ra để bảo đảm tương thích với các mô-đun CoTMix phía sau. Nếu mô hình có lớp chiếu hoặc bộ phân loại, cần xác định chức năng của các thành phần này và cách chúng chuyển biểu diễn đặc trưng thành đầu ra phục vụ huấn luyện hoặc phân loại. Cuối cùng, cần xác minh bộ mã hóa sử dụng trọng số đã được huấn luyện trước hay được khởi tạo mới, cũng như cách các trọng số được cập nhật trong quá trình huấn luyện.

Không nên kết luận rằng tất cả các thành phần nêu trên đều đã được thay đổi. Chỉ sau khi đối chiếu mã nguồn mới có thể xác định thành phần nào được giữ nguyên, thành phần nào được thay thế và thành phần nào được bổ sung.

### 2.3. Lý do cần điều chỉnh bộ mã hóa

Việc sử dụng bộ mã hóa có nguồn gốc từ mô hình thị giác cho dữ liệu cảm biến đặt ra ba yêu cầu chính. Thứ nhất, dữ liệu đầu vào phải có cấu trúc phù hợp với cách bộ mã hóa tiếp nhận và xử lý thông tin. Thứ hai, biểu diễn đặc trưng đầu ra phải tương thích với các thành phần tiếp theo trong CoTMix. Thứ ba, cách khởi tạo và cập nhật trọng số phải phù hợp với mục tiêu nhận dạng hoạt động. Đây là những yêu cầu về thiết kế cần xem xét; chúng không thay thế việc xác minh các thay đổi đã được thực hiện trong mã nguồn.

## 3. Quy trình xử lý của dự án

Quy trình tổng quát bắt đầu từ dữ liệu UCI-HAR ban đầu. Dữ liệu cảm biến được chuẩn bị và chia theo từng người tham gia, sau đó được lưu thành các tệp `.pt` trong thư mục `CoTMix/data/HAR/`. Các tệp này được đọc thông qua bộ dữ liệu và bộ nạp dữ liệu của mô hình. Dữ liệu tiếp tục đi qua bộ mã hóa được tích hợp trong dự án để tạo biểu diễn đặc trưng. Những biểu diễn này được đưa vào các mô-đun CoTMix và bộ phân loại phục vụ quá trình huấn luyện hoặc đánh giá.

**Sơ đồ quy trình xử lý:**

```text
UCI-HAR raw dataset
        |
        v
Chuẩn bị dữ liệu signal
        |
        v
Chia dữ liệu theo subject
        |
        v
CoTMix/data/HAR/
  train_ID.pt
  test_ID.pt
        |
        v
Dataset / DataLoader
        |
        v
Encoder tích hợp trong dự án
        |
        v
Biểu diễn đặc trưng
        |
        v
Các mô-đun CoTMix và đầu phân loại
        |
        v
Huấn luyện / đánh giá
```

Sơ đồ trên thể hiện luồng xử lý ở mức khái quát. Thứ tự gọi hàm, cách CoTMix sử dụng biểu diễn đặc trưng và cách cấu hình bộ mã hóa cần được xác nhận từ mã nguồn hiện tại của dự án.

### 3.1. Định dạng dữ liệu đầu vào

Mỗi tệp `.pt` chứa một cấu trúc dữ liệu gồm hai trường là `samples` và `labels`. Trường `samples` lưu tensor kiểu `torch.float32` có kích thước `[N, 9, 128]`, còn trường `labels` lưu tensor kiểu `torch.int64` có kích thước `[N]`. Trong đó, `N` là số cửa sổ dữ liệu của người tham gia trong tập dữ liệu tương ứng.

Chín kênh cảm biến gồm `body_acc_x`, `body_acc_y`, `body_acc_z`, `body_gyro_x`, `body_gyro_y`, `body_gyro_z`, `total_acc_x`, `total_acc_y` và `total_acc_z`. Nhãn hoạt động ban đầu của UCI-HAR nằm trong khoảng từ `1` đến `6` và được chuyển về khoảng từ `0` đến `5` để phù hợp với định dạng nhãn được sử dụng trong dự án.

### 3.2. Cấu hình và điểm bắt đầu chương trình

Để xác định cách chạy thực tế, cần kiểm tra cấu hình thí nghiệm, trong đó có bộ dữ liệu, kích thước lô dữ liệu, tốc độ học và số vòng huấn luyện. Đồng thời, cần kiểm tra phần định nghĩa bộ dữ liệu và bộ nạp dữ liệu để biết cách chương trình đọc các tệp `train_ID.pt` và `test_ID.pt`. Phần mã nguồn của bộ mã hóa cần được xem xét để xác định cấu trúc mạng chính, cách tạo biểu diễn đặc trưng và phương thức khởi tạo trọng số.

Ngoài ra, cần xác định tệp khởi chạy quá trình huấn luyện, nơi mô hình, bộ tối ưu và vòng lặp huấn luyện được thiết lập. Tệp khởi chạy đánh giá cũng cần được kiểm tra để biết cách tải mô hình đã lưu và đánh giá trên tập kiểm thử. Khi chưa kiểm tra cấu trúc mã nguồn hiện tại, không nên tự giả định tên tệp cấu hình hoặc câu lệnh chạy cụ thể.

## 4. Chuyển đổi dữ liệu UCI-HAR sang định dạng `.pt`

### 4.1. Cấu trúc dữ liệu ban đầu

Tập dữ liệu đầu vào mà chương trình chuyển đổi được thiết kế để sử dụng có cấu trúc như sau:

```text
UCI-HAR Dataset/
├── train/
│   ├── Inertial Signals/
│   │   ├── body_acc_x_train.txt
│   │   ├── body_acc_y_train.txt
│   │   ├── body_acc_z_train.txt
│   │   ├── body_gyro_x_train.txt
│   │   ├── body_gyro_y_train.txt
│   │   ├── body_gyro_z_train.txt
│   │   ├── total_acc_x_train.txt
│   │   ├── total_acc_y_train.txt
│   │   └── total_acc_z_train.txt
│   ├── subject_train.txt
│   └── y_train.txt
└── test/
    ├── Inertial Signals/
    │   └── ... các tệp tín hiệu có hậu tố _test.txt
    ├── subject_test.txt
    └── y_test.txt
```

Thư mục `Inertial Signals` chứa các tệp tín hiệu của chín kênh cảm biến. Tệp `subject_train.txt` hoặc `subject_test.txt` lưu mã người tham gia tương ứng với từng mẫu, còn `y_train.txt` hoặc `y_test.txt` lưu nhãn hoạt động.

### 4.2. Thiết lập đường dẫn

Trong tệp `prepare_uci_har_cotmix.py`, đường dẫn đến dữ liệu đầu vào và thư mục lưu kết quả được khai báo trực tiếp ở phần đầu tệp. Cấu hình hiện tại như sau:

```python
DATASET_ROOT = Path(
    r"D:\Bob\TaiLieuHoc\ChuyenDe\CoTMix_DINOv3_UCI_HAR\UCI-HAR Dataset"
)

OUTPUT_DIR = Path(
    r"D:\Bob\TaiLieuHoc\ChuyenDe\CoTMix_DINOv3_UCI_HAR\CoTMix\data\HAR"
)

TEST_SIZE = 0.2
RANDOM_SEED = 42
SUBJECTS = "all"
```

`DATASET_ROOT` xác định vị trí của bộ dữ liệu UCI-HAR, trong khi `OUTPUT_DIR` xác định thư mục lưu các tệp sau chuyển đổi. `TEST_SIZE = 0.2` đặt tỷ lệ tập kiểm thử là 20%, `RANDOM_SEED = 42` xác định giá trị khởi tạo cho thao tác chia dữ liệu có yếu tố ngẫu nhiên, còn `SUBJECTS = "all"` cho biết chương trình xử lý tất cả người tham gia có dữ liệu. Nếu thay đổi vị trí của dự án hoặc bộ dữ liệu, cần cập nhật hai đường dẫn tương ứng.

### 4.3. Cài đặt thư viện

Nếu dự án chưa có môi trường đã cài đặt các thư viện cần thiết, có thể mở PowerShell tại thư mục dự án và chạy lệnh sau:

```powershell
python -m pip install numpy torch scikit-learn
```

Nếu dự án đã có môi trường ảo hoặc tệp khai báo thư viện phụ thuộc, nên kích hoạt môi trường đó và cài đặt theo cấu hình hiện có để bảo đảm tính nhất quán.

### 4.4. Thực hiện chuyển đổi dữ liệu

Tại thư mục chứa tệp `prepare_uci_har_cotmix.py`, chạy lệnh sau để bắt đầu quá trình chuyển đổi:

```powershell
python -u .\prepare_uci_har_cotmix.py
```

Tệp chương trình có hàm `main()` và sử dụng đoạn mã sau làm điểm bắt đầu thực thi:

```python
if __name__ == "__main__":
    main()
```

Khi được thực thi, chương trình kiểm tra đường dẫn bộ dữ liệu và sự tồn tại của các tệp cần thiết, sau đó đọc chín kênh tín hiệu trong hai phần dữ liệu ban đầu là `train` và `test`. Chương trình tiếp tục đọc mã người tham gia cùng nhãn hoạt động tương ứng, chuyển nhãn từ khoảng `1..6` sang `0..5`, rồi kết hợp dữ liệu của hai phần. Các cửa sổ dữ liệu được nhóm theo từng người tham gia và mỗi nhóm được chia thành tập huấn luyện, tập kiểm thử theo tỷ lệ 80/20. Cuối cùng, dữ liệu được lưu thành các tệp `.pt` trong thư mục đầu ra đã cấu hình.

### 4.5. Dữ liệu sau chuyển đổi

Sau khi hoàn tất, thư mục đầu ra dự kiến có cấu trúc như sau:

```text
CoTMix/
└── data/
    └── HAR/
        ├── train_1.pt
        ├── test_1.pt
        ├── train_2.pt
        ├── test_2.pt
        ├── ...
        ├── train_30.pt
        └── test_30.pt
```

Tên và số lượng tệp thực tế phụ thuộc vào những người tham gia có trong dữ liệu và giá trị của `SUBJECTS`. Người tham gia không có mẫu dữ liệu sẽ không tạo ra tệp tương ứng.

### 4.6. Kiểm tra tệp dữ liệu sau chuyển đổi

Sau khi chuyển đổi, có thể dùng đoạn mã sau để kiểm tra cấu trúc và kiểu dữ liệu của một tệp:

```python
from pathlib import Path
import torch

path = Path("data/HAR/train_1.pt")
data = torch.load(path, map_location="cpu", weights_only=True)

print(data.keys())
print(data["samples"].shape)
print(data["samples"].dtype)
print(data["labels"].shape)
print(data["labels"].dtype)
print(data["labels"].min().item())
print(data["labels"].max().item())
```

Kết quả kiểm tra cần cho thấy tệp có hai trường `samples` và `labels`. Tensor `samples` phải có kích thước `[N, 9, 128]` và kiểu `torch.float32`; tensor `labels` phải có kích thước `[N]` và kiểu `torch.int64`. Giá trị nhãn nhỏ nhất và lớn nhất cần nằm trong khoảng từ `0` đến `5`. Giá trị `N` có thể khác nhau giữa các người tham gia cũng như giữa tập huấn luyện và tập kiểm thử.

## 5. Lưu ý về phương pháp chia dữ liệu và đánh giá

Phương pháp chia dữ liệu trong chương trình chuyển đổi không trùng với phương pháp chia tập huấn luyện và tập kiểm thử ban đầu của UCI-HAR. Trong bộ dữ liệu gốc, người tham gia được phân chia giữa tập huấn luyện và tập kiểm thử. Trong khi đó, chương trình của dự án kết hợp các bản ghi từ hai phần ban đầu, sau đó chia lại các cửa sổ dữ liệu của từng người tham gia thành hai tập theo tỷ lệ 80/20.

Cách chia này tạo ra cả tệp `train_ID.pt` và `test_ID.pt` cho từng người tham gia, phù hợp với định dạng dữ liệu mà cấu hình CoTMix của dự án đang hướng tới. Tuy nhiên, kết quả thu được từ cách chia này không thể được xem là kết quả đánh giá theo phương pháp chia chính thức của UCI-HAR. Khi trình bày kết quả thực nghiệm, cần mô tả rõ cách chia dữ liệu đã sử dụng để tránh so sánh trực tiếp các kết quả được tạo ra từ những phương pháp đánh giá khác nhau.

## 6. Kiểm tra trước khi huấn luyện

Trước khi bắt đầu huấn luyện, cần xác nhận rằng `DATASET_ROOT` trỏ đúng đến thư mục chứa `train/` và `test/`, đồng thời các tệp `.pt` đã được tạo trong `CoTMix/data/HAR/`. Cần kiểm tra kích thước tensor, kiểu dữ liệu và phạm vi nhãn để bảo đảm dữ liệu đầu ra đúng định dạng. Cấu hình dữ liệu của CoTMix cũng phải trỏ đến đúng thư mục và tên tệp.

Bên cạnh đó, cần xác nhận bộ mã hóa tiếp nhận đúng kích thước đầu vào và tạo ra biểu diễn đặc trưng có kích thước phù hợp với các mô-đun phía sau. Phương thức tải trọng số đã huấn luyện trước, cũng như việc cố định hoặc tiếp tục cập nhật các trọng số của bộ mã hóa, cần được kiểm tra từ mã nguồn và cấu hình thực tế. Cuối cùng, cần xác minh câu lệnh huấn luyện, câu lệnh đánh giá và ghi rõ phương pháp chia dữ liệu được sử dụng trong báo cáo kết quả.

## 7. Những nội dung cần xác minh để hoàn thiện báo cáo

Để mô tả chính xác những thay đổi của bộ mã hóa, cần đối chiếu mã nguồn ban đầu với phiên bản đang được sử dụng trong dự án. Việc đối chiếu cần xác định những lớp hoặc mô-đun đã được thêm, loại bỏ hoặc thay thế; cách tensor `[B, 9, 128]` được chuyển thành đầu vào của mạng chính; và kích thước, cấu trúc của biểu diễn đặc trưng đầu ra. Đồng thời, cần làm rõ những thành phần nào của DINOv3 được giữ lại, những thành phần nào được huấn luyện lại và cách bộ mã hóa được gọi trong mô hình CoTMix.

Cần kiểm tra thêm tệp cấu hình và câu lệnh thực tế được sử dụng trong thí nghiệm. Khi đã xác minh đầy đủ các thông tin này, tài liệu có thể được bổ sung phần so sánh giữa bộ mã hóa gốc và bộ mã hóa đã điều chỉnh, cùng sơ đồ thể hiện mối liên hệ giữa các mô-đun theo đúng mã nguồn. Cho đến khi có các bằng chứng từ mã nguồn, những nội dung này cần được xem là các điểm cần xác minh thay vì mô tả về những thay đổi đã được xác nhận.
