# CoTMix kết hợp DINOv3 trên bộ dữ liệu UCI-HAR

## 1. Giới thiệu

Báo cáo này trình bày cách tích hợp DINOv3 vào quy trình huấn luyện CoTMix cho bài toán nhận dạng hoạt động của con người trên bộ dữ liệu UCI-HAR. Dữ liệu UCI-HAR được thu thập từ các cảm biến quán tính gắn trên cơ thể, vì vậy đầu vào của mô hình là các chuỗi tín hiệu thay đổi theo thời gian. Mục tiêu của dự án là chuẩn bị dữ liệu theo định dạng mà CoTMix sử dụng, thay bộ mã hóa đặc trưng ban đầu bằng một bộ mã hóa dựa trên DINOv3, sau đó thực hiện huấn luyện và đánh giá theo quy trình của repository.

Điểm khác biệt quan trọng giữa hai mô hình nằm ở loại dữ liệu mà bộ mã hóa được thiết kế để xử lý. Bộ mã hóa trong CoTMix hướng đến tín hiệu chuỗi thời gian, trong khi DINOv3 được xây dựng chủ yếu để học đặc trưng từ hình ảnh. Vì vậy, không thể xem việc tích hợp DINOv3 là thao tác thay trực tiếp một mạng bằng mạng khác. Cần có một bước chuyển đổi để đưa tín hiệu cảm biến về dạng đầu vào phù hợp với mô hình thị giác, đồng thời bảo đảm đầu ra của bộ mã hóa vẫn kết nối được với các thành phần còn lại của CoTMix.

Tài liệu mô tả mục tiêu thiết kế và quy trình dữ liệu của phiên bản thử nghiệm. Những chi tiết cụ thể như tên lớp đã thay đổi, cách lấy đặc trưng, kích thước đầu ra và trạng thái đóng băng trọng số cần được xác nhận bằng mã nguồn thực tế của repository đang sử dụng trước khi xem là kết luận cuối cùng về implementation.

## 2. Sự khác biệt giữa bộ mã hóa CoTMix và DINOv3

Bộ mã hóa ban đầu của CoTMix được thiết kế để xử lý tín hiệu có thứ tự theo thời gian. Với UCI-HAR, mỗi mẫu gồm chín kênh cảm biến, mỗi kênh có 128 bước thời gian. Dữ liệu theo lô thường có dạng `[B, 9, 128]`, trong đó `B` là số mẫu trong một lô, `9` là số kênh và `128` là độ dài cửa sổ thời gian. Mạng tích chập một chiều có thể học các mẫu biến thiên cục bộ dọc theo trục thời gian, chẳng hạn sự thay đổi gia tốc hoặc vận tốc góc trong một khoảng ngắn. Kiểu biểu diễn này phù hợp với bản chất chuỗi thời gian của tín hiệu cảm biến.

Ngược lại, DINOv3 là mô hình học biểu diễn thị giác được huấn luyện trước trên dữ liệu hình ảnh. Kiến trúc thị giác dạng Transformer thường tiếp nhận ảnh có các kênh màu và chiều cao, chiều rộng xác định. Vì vậy, tensor tín hiệu `[B, 9, 128]` không thể được đưa trực tiếp vào DINOv3 theo cách thông thường mà không điều chỉnh kiến trúc hoặc chuyển đổi dữ liệu. Nếu đưa dữ liệu vào sai định dạng, kích thước và ý nghĩa của các chiều đầu vào sẽ không phù hợp với những gì mô hình được thiết kế để xử lý.

Trong phiên bản thử nghiệm này, hướng triển khai là chuyển tín hiệu cảm biến thành một biểu diễn dạng ảnh trước khi đưa vào DINOv3. Nhờ đó, DINOv3 có thể xử lý biểu diễn tín hiệu thông qua đường xử lý ảnh của nó. Cách làm này tạo ra một cầu nối giữa dữ liệu chuỗi thời gian và mô hình thị giác, nhưng cũng làm thay đổi cách thông tin ban đầu được biểu diễn. Vì vậy, bước chuyển đổi phải được xem là một thành phần quan trọng của mô hình, không chỉ là thao tác định dạng dữ liệu.

## 3. Thay đổi trong bộ mã hóa và lý do thiết kế

Thay đổi trọng tâm là sử dụng DINOv3 làm bộ mã hóa đặc trưng thay cho bộ mã hóa tín hiệu ban đầu của CoTMix. Tuy nhiên, DINOv3 không trực tiếp thay thế toàn bộ quy trình huấn luyện CoTMix. Các thành phần xử lý thích nghi miền và phân loại vẫn cần được kết nối với đặc trưng đầu ra của bộ mã hóa mới. Vì vậy, phần tích hợp cần xử lý cả đầu vào của DINOv3 lẫn sự tương thích giữa đặc trưng đầu ra và bộ phân loại.

Bước chuyển tín hiệu thành ảnh là cần thiết vì DINOv3 được thiết kế cho ảnh, còn tín hiệu UCI-HAR là chuỗi số theo thời gian. Một phương án có thể dùng là biểu diễn tín hiệu bằng trường góc Gramian (Gramian Angular Field, GAF). Phương pháp này chuyển quan hệ giữa các thời điểm trong chuỗi thành một ma trận hai chiều, nhờ đó tín hiệu có thể được biểu diễn dưới dạng ảnh. Nếu triển khai sử dụng GAF, cần xác nhận trong mã nguồn cách chuẩn hóa tín hiệu, cách tạo ma trận GAF và cách kết hợp chín kênh thành đầu vào có ba kênh màu. Không nên mặc định các bước này đã có trong chương trình nếu chưa kiểm tra trực tiếp lớp mã hóa.

Sau bước chuyển đổi, ảnh được đưa vào DINOv3 để tạo biểu diễn đặc trưng. Bộ phân loại phía sau phải nhận đúng kích thước đặc trưng mà DINOv3 trả về. Đây là lý do phần tích hợp thường cần điều chỉnh cách khởi tạo bộ phân loại: thay vì giả định đầu ra có kích thước cố định theo mạng tích chập một chiều ban đầu, chương trình cần xác định số chiều đặc trưng từ bộ mã hóa đang được chọn. Nếu kích thước đầu vào bộ phân loại không khớp với đầu ra của DINOv3, mô hình sẽ phát sinh lỗi về kích thước tensor khi thực hiện lan truyền xuôi.

Việc dùng trọng số đã huấn luyện trước của DINOv3 có thể tận dụng biểu diễn đã học từ dữ liệu hình ảnh lớn. Tuy nhiên, lợi ích này không bảo đảm rằng đặc trưng đó sẽ phù hợp tự động với dữ liệu cảm biến. Quá trình chuyển tín hiệu thành ảnh có thể làm mất hoặc thay đổi một phần thông tin thời gian, tùy cách xây dựng biểu diễn. Vì vậy, cần đánh giá bằng thực nghiệm và so sánh với bộ mã hóa ban đầu của CoTMix trên cùng giao thức dữ liệu. Cũng cần ghi lại bộ mã hóa có được đóng băng hay được tinh chỉnh trong quá trình huấn luyện, bởi lựa chọn này ảnh hưởng đến thời gian chạy, mức sử dụng bộ nhớ và kết quả mô hình.

## 4. Cấu trúc xử lý hiện tại

Luồng dữ liệu tổng quát bắt đầu từ bộ dữ liệu UCI-HAR ở dạng tín hiệu thô. Script chuẩn bị dữ liệu đọc chín kênh cảm biến, nhãn hoạt động và mã người tham gia, sau đó lưu các mẫu thành file `.pt`. DataLoader của CoTMix đọc các file này và tạo các lô dữ liệu để đưa vào quy trình huấn luyện. Ở bước mô hình, tín hiệu được chuyển thành biểu diễn dạng ảnh trước khi DINOv3 trích xuất đặc trưng. Đặc trưng sau đó được chuyển đến bộ phân loại và các thành phần học của CoTMix.

```text
UCI-HAR dạng tín hiệu
        |
        v
Script chuyển đổi dữ liệu
        |
        v
Các file .pt theo subject
        |
        v
DataLoader của CoTMix
        |
        v
Tensor tín hiệu [B, 9, 128]
        |
        v
Chuyển tín hiệu thành biểu diễn ảnh
        |
        v
DINOv3 — bộ mã hóa đặc trưng
        |
        v
Vector đặc trưng
        |
        v
Bộ phân loại và các thành phần CoTMix
        |
        v
Huấn luyện và đánh giá
```

Sơ đồ trên thể hiện luồng xử lý về mặt khái niệm. Trình tự gọi hàm chính xác và vị trí thực hiện từng bước cần được đối chiếu với các file `main.py`, `trainer.py`, `algorithms.py`, `dataloader.py` và các file trong thư mục `models/`. Khi báo cáo kết quả, cần phân biệt rõ sơ đồ thiết kế với luồng đã được xác minh trong mã nguồn.

## 5. Định dạng dữ liệu `.pt` sử dụng trong CoTMix

Mỗi file `.pt` được tạo ra chứa một từ điển với hai trường là `samples` và `labels`. Trường `samples` là tensor `torch.float32` có dạng `[N, 9, 128]`, trong đó `N` là số cửa sổ tín hiệu của subject trong tập dữ liệu tương ứng. Trường `labels` là tensor `torch.int64` có dạng `[N]`. Nhãn hoạt động gốc của UCI-HAR nằm trong khoảng từ 1 đến 6, vì vậy script chuyển nhãn sang khoảng từ 0 đến 5 để phù hợp với cách đánh số lớp trong bài toán phân loại.

Chín kênh cảm biến gồm ba trục gia tốc cơ thể (`body_acc_x`, `body_acc_y`, `body_acc_z`), ba trục vận tốc góc (`body_gyro_x`, `body_gyro_y`, `body_gyro_z`) và ba trục gia tốc tổng (`total_acc_x`, `total_acc_y`, `total_acc_z`). Mỗi kênh trong dữ liệu tín hiệu thô cần có 128 giá trị cho mỗi cửa sổ. Nếu bộ dữ liệu tải từ Kaggle chỉ có các đặc trưng đã trích xuất, chẳng hạn ma trận gồm 561 đặc trưng, thì không thể sử dụng trực tiếp script dành cho thư mục `Inertial Signals`; cần xác định đúng loại dữ liệu trước khi chuyển đổi.

## 6. Chuyển đổi dữ liệu UCI-HAR từ signal sang `.pt`

### 6.1. Cấu trúc dữ liệu đầu vào

Script `prepare_uci_har_cotmix.py` được thiết kế cho bộ dữ liệu có thư mục `train/` và `test/`. Mỗi thư mục cần có thư mục con `Inertial Signals/` chứa chín file tín hiệu. Tập huấn luyện cần có các file `body_acc_x_train.txt`, `body_acc_y_train.txt`, `body_acc_z_train.txt`, `body_gyro_x_train.txt`, `body_gyro_y_train.txt`, `body_gyro_z_train.txt`, `total_acc_x_train.txt`, `total_acc_y_train.txt` và `total_acc_z_train.txt`, cùng với `subject_train.txt` và `y_train.txt`. Tập kiểm thử cần các file tương ứng có hậu tố `_test.txt`, cùng `subject_test.txt` và `y_test.txt`.

Cấu trúc thư mục dự kiến như sau:

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
    │   └── Các file tín hiệu có hậu tố _test.txt
    ├── subject_test.txt
    └── y_test.txt
```

### 6.2. Cấu hình đường dẫn

Đường dẫn dữ liệu đầu vào và thư mục lưu kết quả được khai báo trực tiếp ở phần cấu hình gần đầu file `prepare_uci_har_cotmix.py`. Với cấu trúc thư mục đang dùng trong dự án, cấu hình có dạng:

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

`DATASET_ROOT` phải trỏ đến thư mục chứa `train/` và `test/`, còn `OUTPUT_DIR` là nơi lưu các file `.pt`. Nếu chuyển repository hoặc dữ liệu sang vị trí khác, cần sửa hai đường dẫn này. Giá trị `TEST_SIZE = 0.2` đặt tỷ lệ kiểm thử là 20% khi chia lại dữ liệu trong từng subject. `RANDOM_SEED = 42` giúp tái tạo cách chia dữ liệu trong các lần chạy sau. `SUBJECTS = "all"` yêu cầu xử lý tất cả subject có trong dữ liệu.

### 6.3. Cài đặt và chạy script

Script cần NumPy, PyTorch và scikit-learn. Có thể cài các thư viện này trong môi trường Python đang dùng cho dự án bằng lệnh:

```powershell
python -m pip install numpy torch scikit-learn
```

Mở PowerShell tại thư mục gốc của repository, nơi đặt file `prepare_uci_har_cotmix.py`, rồi chạy:

```powershell
python -u .\prepare_uci_har_cotmix.py
```

Nếu sử dụng Python Launcher trên Windows và muốn chỉ định Python 3.13, có thể chạy:

```powershell
py -3.13 .\prepare_uci_har_cotmix.py
```

Khi chạy, script kiểm tra đường dẫn và các file cần thiết, đọc dữ liệu của chín kênh tín hiệu, đọc nhãn và mã subject, sau đó ghép các bản ghi từ hai partition gốc. Dữ liệu được nhóm theo subject, chia thành tập huấn luyện và kiểm thử theo tỷ lệ đã cấu hình, rồi lưu thành các file `.pt`. Nếu cấu trúc dữ liệu không đúng định dạng yêu cầu, script sẽ báo đường dẫn hoặc file bị thiếu để người dùng kiểm tra.

### 6.4. Cấu trúc đầu ra và kiểm tra

Các file được lưu trong `CoTMix/data/HAR/` theo quy tắc tên `train_ID.pt` và `test_ID.pt`, trong đó `ID` là mã subject. Ví dụ, thư mục có thể chứa `train_1.pt`, `test_1.pt`, `train_2.pt` và `test_2.pt`. Danh sách file thực tế phụ thuộc vào các subject có trong dữ liệu và cấu hình `SUBJECTS`.

Có thể kiểm tra định dạng một file bằng đoạn mã sau:

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

Kết quả cần có hai trường `samples` và `labels`. `samples` phải có kiểu `torch.float32` và dạng `[N, 9, 128]`; `labels` phải có kiểu `torch.int64` và dạng `[N]`, với nhãn trong khoảng từ 0 đến 5. Số lượng mẫu `N` có thể khác nhau giữa các subject và giữa tập huấn luyện với tập kiểm thử.

## 7. Cách chia dữ liệu và giới hạn đánh giá

Cách chia dữ liệu của script hiện tại không giống cách chia chính thức của UCI-HAR. Trong bộ dữ liệu gốc, tập huấn luyện và tập kiểm thử được phân chia theo người tham gia. Script chuyển đổi này kết hợp dữ liệu từ cả hai partition, sau đó chia ngẫu nhiên các cửa sổ trong từng subject thành tập huấn luyện và tập kiểm thử. Cách làm này tạo ra cả `train_ID.pt` và `test_ID.pt` cho mỗi subject, phù hợp với cấu trúc file theo subject mà cấu hình CoTMix đang hướng tới.

Tuy nhiên, vì các cửa sổ của cùng một subject có thể xuất hiện ở cả hai tập sau khi chia lại, kết quả không thể được xem là kết quả theo giao thức đánh giá chính thức của UCI-HAR. Khi so sánh mô hình hoặc viết báo cáo thực nghiệm, cần ghi rõ cách chia dữ liệu, danh sách subject, giá trị seed và cách lựa chọn miền nguồn, miền đích. Các kết quả dùng giao thức khác nhau không nên được so sánh trực tiếp nếu không giải thích sự khác biệt.

## 8. Kiểm tra trước khi huấn luyện

Trước khi bắt đầu huấn luyện, cần xác nhận `DATASET_ROOT` trỏ đúng thư mục dữ liệu và các file `.pt` đã được tạo trong `CoTMix/data/HAR/`. Nên kiểm tra shape, kiểu dữ liệu và phạm vi nhãn của một số file để phát hiện lỗi chuyển đổi sớm. Cấu hình dataset trong CoTMix cũng cần trỏ đến đúng thư mục và sử dụng đúng quy tắc tên file mà DataLoader mong đợi.

Đối với encoder, cần xác minh rằng bước chuyển tín hiệu thành ảnh được gọi đúng vị trí, đầu vào sau chuyển đổi có định dạng mà DINOv3 chấp nhận, và embedding đầu ra có kích thước phù hợp với bộ phân loại. Cũng cần kiểm tra checkpoint pretrained, trạng thái đóng băng hoặc tinh chỉnh trọng số và cách khởi tạo bộ tối ưu. Những thông tin này nên được ghi lại cùng kết quả thí nghiệm để bảo đảm khả năng tái lập.

## 9. Kết luận

Việc kết hợp DINOv3 với CoTMix cho UCI-HAR đòi hỏi phải giải quyết sự khác biệt giữa dữ liệu chuỗi thời gian và dữ liệu hình ảnh. Bộ mã hóa gốc của CoTMix được thiết kế để học đặc trưng từ tín hiệu cảm biến biến thiên theo thời gian, trong khi DINOv3 được thiết kế chủ yếu để xử lý ảnh. Vì vậy, cần có bước chuyển tín hiệu thành biểu diễn dạng ảnh trước khi đưa dữ liệu vào DINOv3, đồng thời điều chỉnh kết nối với bộ phân loại để phù hợp với đặc trưng đầu ra.

Script chuyển đổi dữ liệu tạo ra các file `.pt` theo subject để DataLoader của CoTMix có thể sử dụng. Tuy nhiên, cách chia lại dữ liệu trong script khác với giao thức chính thức của UCI-HAR và cần được ghi rõ trong báo cáo. Cuối cùng, mọi khẳng định về thay đổi cụ thể trong encoder, phương pháp chuyển tín hiệu thành ảnh và cấu hình huấn luyện cần được xác nhận từ mã nguồn của phiên bản đang chạy. Hiệu quả của DINOv3 trong bài toán này phải được đánh giá bằng thực nghiệm, không thể suy ra chỉ từ việc sử dụng một mô hình thị giác đã được huấn luyện trước.
