# Tinh chỉnh loss function cho CoTMix với backbone DINOv3

## 1. Bối cảnh

Dự án áp dụng phương pháp CoTMix (Contrastive Domain Adaptation for Time-Series via Temporal Mixup) cho bài toán Human Activity Recognition trên bộ dữ liệu UCI-HAR, với thiết lập cross-subject domain adaptation: mô hình học trên dữ liệu có nhãn của một người (source domain) và được đánh giá trên một người khác chưa có nhãn (target domain). Thí nghiệm dùng năm cặp kịch bản 2→11, 6→23, 7→13, 9→18 và 12→16, mỗi người có khoảng 240 cửa sổ huấn luyện và khoảng 60 cửa sổ kiểm thử, mỗi cửa sổ gồm 9 kênh cảm biến với 128 mốc thời gian.

Bên cạnh encoder gốc là 1D-CNN, dự án thử nghiệm thay encoder bằng DINOv3 ViT-S/16, một Vision Transformer được pretrain theo hướng self-supervised trên ảnh tự nhiên. Vì DINOv3 nhận đầu vào là ảnh, mỗi cửa sổ tín hiệu được chuyển thành ảnh Gramian Angular Field (GAF) ba kênh kích thước 224×224, sau đó lấy embedding của CLS token (384 chiều) đưa vào một classifier tuyến tính. DINOv3 được giữ ở trạng thái frozen, nghĩa là trong toàn bộ quá trình huấn luyện chỉ có lớp Linear của classifier được cập nhật.

Với loss function gốc của CoTMix, mô hình DINOv3 sau 1 epoch chỉ đạt Accuracy trung bình 28.44% và Macro-F1 21.87%, chỉ cao hơn không nhiều so với mức đoán ngẫu nhiên khoảng 16.7% của bài toán sáu lớp. Mục tiêu của phần tinh chỉnh là sửa loss function để kết quả sau 1 epoch tốt hơn một cách có thể giải thích được, trong khi vẫn giữ nguyên các thiết lập khác như learning rate, batch size, seed và dữ liệu.

## 2. Loss function gốc của CoTMix

Loss gốc là tổng có trọng số của bốn thành phần. Thành phần thứ nhất là Cross-Entropy trên mẫu nguồn với trọng số 0.78. Thành phần thứ hai là Conditional Entropy trên dự đoán của mẫu đích với trọng số 0.05, nhằm khuyến khích mô hình đưa ra dự đoán tự tin trên target domain. Thành phần thứ ba là Supervised Contrastive Loss (SupCon) giữa mẫu nguồn và mẫu trộn nghiêng về nguồn (source-dominant mixup) với trọng số 0.1. Thành phần thứ tư là NT-Xent Contrastive Loss giữa mẫu đích và mẫu trộn nghiêng về đích (target-dominant mixup) với trọng số 0.1.

Mẫu trộn được tạo bởi Temporal Mixup với mix_ratio bằng 0.9, tức là mẫu source-dominant giữ 90% tín hiệu của mẫu nguồn và 10% là trung bình trượt theo thời gian của mẫu đích. Một chi tiết quan trọng là cả SupCon và NT-Xent trong mã nguồn gốc đều được tính trực tiếp trên logits sáu chiều đầu ra của classifier, không phải trên một không gian embedding riêng.

## 3. Phương pháp thử nghiệm

Mỗi lần chạy thật một epoch cho cả năm kịch bản mất khoảng 10 phút trên CPU, chủ yếu do chi phí forward của DINOv3. Để có thể so sánh nhiều biến thể loss một cách nhanh và công bằng, chúng tôi tận dụng tính chất frozen của backbone: vì DINOv3 không được cập nhật, embedding của mỗi batch chỉ phụ thuộc vào dữ liệu đầu vào chứ không phụ thuộc vào loss. Do đó embedding của đúng các batch mà lần chạy thật sử dụng (cùng seed, cùng thứ tự shuffle, cùng mẫu Temporal Mixup) được tính một lần và lưu lại, sau đó mỗi biến thể loss chỉ cần huấn luyện lại classifier tuyến tính trên các embedding đã lưu, mất vài giây cho mỗi lần thử.

Công cụ thử nghiệm này tái tạo chính xác kết quả gốc đến từng kịch bản (Accuracy 28.44%, Macro-F1 21.87%), và cấu hình cuối cùng sau khi đưa vào mã nguồn cũng cho đúng con số mà công cụ dự đoán khi chạy lại bằng main.py. Mọi so sánh trong tài liệu này đều dùng cùng seed, cùng learning rate 1e-3 cho classifier, batch size 16 và một lần chạy cho mỗi kịch bản.

## 4. Chẩn đoán từ log

Log sau 1 epoch của cấu hình gốc cho thấy src_cls_loss vẫn ở mức khoảng 1.72, rất sát giá trị ln(6) ≈ 1.79 của một dự đoán phân bố đều trên sáu lớp. Điều này nghĩa là classifier gần như chưa học được ranh giới phân lớp sau 15 bước cập nhật. Trong khi đó trg_con_loss (NT-Xent) ở mức khoảng 2.6, là thành phần có giá trị lớn nhất và vẫn nằm gần mức tối đa ln(2B−1) ≈ 3.43 với B = 16.

Từ đó có thể nhận định rằng ở giai đoạn đầu, khi logits còn gần như ngẫu nhiên, các auxiliary loss của CoTMix tác động trực tiếp lên chính những logits chưa được huấn luyện. NT-Xent ép logits của từng mẫu đích phải giống logits của mẫu trộn tương ứng và khác các mẫu còn lại trong batch, một mục tiêu không liên quan đến nhãn lớp và cạnh tranh gradient với Cross-Entropy. Với backbone frozen, lớp Linear là tham số duy nhất có thể thay đổi, nên mọi gradient lệch hướng đều trực tiếp làm chậm việc học phân lớp. Quan sát tương tự cũng xuất hiện ở log của 1D-CNN sau 40 epoch: phần đóng góp của hai contrastive loss vào loss tổng (khoảng 0.12 và 0.22) lớn hơn phần đóng góp của Cross-Entropy (khoảng 0.11), trong khi giá trị của chúng gần như không giảm qua các epoch.

## 5. Các điều chỉnh loss function

Điều chỉnh thứ nhất là warmup cho target contrastive loss. NT-Xent trên target domain được tắt trong epoch đầu tiên và bật lại với trọng số gốc 0.1 từ epoch thứ hai. Cách làm này giữ nguyên thiết kế của CoTMix về lâu dài nhưng cho classifier một khoảng thời gian học từ tín hiệu có nhãn trước khi bắt đầu căn chỉnh miền. Khi thử riêng lẻ, việc bỏ NT-Xent trong epoch đầu nâng Accuracy từ 28.44% lên 37.92% và Macro-F1 từ 21.87% lên 28.40%, là thay đổi đơn lẻ có tác động lớn nhất.

Điều chỉnh thứ hai là bổ sung Source-mix Cross-Entropy, tức là Cross-Entropy trên logits của mẫu source-dominant với nhãn của mẫu nguồn, trọng số 0.78 bằng với Cross-Entropy gốc. Vì mix_ratio bằng 0.9, mẫu trộn vẫn mang 90% tín hiệu nguồn nên nhãn nguồn vẫn hợp lệ, và việc này có thể xem như một dạng data augmentation có giám sát. Logits của mẫu source-dominant vốn đã được CoTMix tính sẵn để phục vụ SupCon, nên thành phần này tăng gấp đôi lượng tín hiệu có nhãn mà không cần thêm lượt forward nào qua DINOv3. Khi thử riêng lẻ, thành phần này nâng Accuracy lên 34.85% và Macro-F1 lên 26.48%.

Điều chỉnh thứ ba là bổ sung Diversity Loss theo tinh thần Information Maximization, với trọng số 0.3. Loss này bằng âm entropy của phân phối dự đoán trung bình trên batch đích, −H(p̄), và đạt giá trị nhỏ nhất −ln(6) ≈ −1.79 khi các lớp được dự đoán đều nhau. Conditional Entropy gốc chỉ khuyến khích từng mẫu được dự đoán tự tin, nên có nguy cơ đẩy toàn bộ mẫu đích về một hoặc hai lớp sai (prediction collapse); Diversity Loss đóng vai trò đối trọng cho rủi ro đó. Sau 1 epoch thành phần này có tác động nhỏ, nhưng sau 3 epoch nó giúp Macro-F1 tăng thêm khoảng 1.7 điểm so với cấu hình không có nó.

## 6. Loss function sau tinh chỉnh

Loss function cuối cùng cho DINOv3 có dạng:

```
L = 0.78·CE(source) + 0.78·CE(source-dominant, y_source)
  + 0.05·H(target) + 0.3·[−H(p̄_target)]
  + 0.1·SupCon(source, source-dominant)
  + w(epoch)·0.1·NT-Xent(target, target-dominant)
```

trong đó w(epoch) bằng 0 ở epoch đầu tiên và bằng 1 từ epoch thứ hai trở đi. Ba thành phần mới mặc định có trọng số bằng 0 và chỉ được bật khi chọn backbone DINOv3, nên cấu hình 1D-CNN giữ nguyên hoàn toàn loss function gốc của CoTMix.

## 7. Kết quả

Bảng dưới so sánh kết quả sau 1 epoch giữa loss gốc và loss sau tinh chỉnh, với cùng seed, learning rate 1e-3 và batch size 16.

| Kịch bản | Accuracy gốc | Accuracy mới | Macro-F1 gốc | Macro-F1 mới |
|---|---|---|---|---|
| 12 → 16 | 17.81 | 26.03 | 10.31 | 19.71 |
| 2 → 11 | 26.98 | 39.68 | 17.53 | 25.33 |
| 6 → 23 | 33.78 | 35.14 | 27.73 | 22.50 |
| 7 → 13 | 30.77 | 49.23 | 23.16 | 44.81 |
| 9 → 18 | 32.88 | 43.84 | 30.64 | 36.44 |
| Trung bình | 28.44 | 38.78 | 21.87 | 29.76 |

Accuracy trung bình tăng 10.34 điểm và Macro-F1 trung bình tăng 7.89 điểm. Accuracy tăng ở cả năm kịch bản, trong đó kịch bản 7→13 tăng mạnh nhất với 18.46 điểm Accuracy và 21.65 điểm Macro-F1. Riêng kịch bản 6→23 có Accuracy tăng nhẹ nhưng Macro-F1 giảm 5.23 điểm, cho thấy ở kịch bản này mô hình dự đoán đúng thêm một số mẫu nhưng phân bố dự đoán giữa các lớp kém cân bằng hơn.

Để kiểm tra rằng cải thiện không chỉ là hiện tượng của epoch đầu, cả hai cấu hình được huấn luyện thêm đến 3 epoch. Với loss gốc, kết quả đạt Accuracy 43.36% và Macro-F1 40.60%; với loss sau tinh chỉnh, kết quả đạt Accuracy 47.93% và Macro-F1 43.66%. Như vậy loss mới vẫn duy trì lợi thế khoảng 4.6 điểm Accuracy và 3.1 điểm Macro-F1 khi target contrastive loss đã được bật lại.

Log của epoch đầu với loss mới, ở kịch bản 2→11, ghi nhận src_cls_loss bằng 1.7193, src_mix_cls_loss bằng 1.7328 và trg_div_loss bằng −1.7788. Giá trị trg_div_loss rất gần −ln(6) cho thấy dự đoán trên target vẫn phân bố khá đều giữa các lớp, tức là không xảy ra prediction collapse. Do có thêm hai thành phần, Total_loss của cấu hình mới không còn so sánh trực tiếp được với log cũ; việc đánh giá nên dựa trên từng thành phần loss và trên Accuracy, Macro-F1.

## 8. Các biến thể đã thử nhưng không được chọn

Label Smoothing với hệ số 0.1 cho kết quả gần như không đổi (Accuracy 27.76%, Macro-F1 21.89%), vì sau 1 epoch mô hình chưa đủ tự tin để hiện tượng overconfidence xuất hiện. Chuẩn hóa L2 logits trước khi tính SupCon làm kết quả giảm xuống Accuracy 26.35% và Macro-F1 20.47%. Bỏ hoàn toàn Conditional Entropy cho Accuracy 27.85% và Macro-F1 21.67%, tức là thành phần này không phải nguyên nhân gây hại. Pseudo-labeling trên target với ngưỡng confidence 0.9 không có tác dụng vì sau 1 epoch không có mẫu đích nào vượt ngưỡng. Tăng temperature của NT-Xent từ 0.2 lên 0.5 thay vì tắt nó cũng có cải thiện (Accuracy 35.78%, Macro-F1 27.89%) nhưng kém hơn phương án warmup.

Cấu hình chỉ gồm warmup NT-Xent và Source-mix Cross-Entropy, không có Diversity Loss, cho kết quả 1 epoch tương đương cấu hình cuối cùng (Accuracy 38.85%, Macro-F1 29.86%). Tuy nhiên sau 3 epoch, cấu hình có Diversity Loss tốt hơn rõ rệt (Macro-F1 43.66% so với 41.94%), nên Diversity Loss được giữ lại trong phiên bản cuối.

## 9. Hạn chế và lưu ý khi diễn giải

Yếu tố ảnh hưởng mạnh nhất đến kết quả sau 1 epoch không phải loss function mà là learning rate của classifier. Chỉ cần tăng learning rate từ 1e-3 lên 1e-2 với loss gốc, Accuracy sau 1 epoch đã đạt 49.93% và Macro-F1 đạt 44.82%. Ở learning rate 1e-2, loss sau tinh chỉnh cho Accuracy 48.80% và Macro-F1 41.25%, thấp hơn loss gốc; phương án tốt nhất ở mức learning rate này chỉ là thêm Diversity Loss với trọng số 0.1, đạt Accuracy 50.53% và Macro-F1 45.37%, một mức chênh nằm trong phạm vi nhiễu. Vì vậy kết luận của tài liệu này chỉ áp dụng cho thiết lập learning rate mặc định 1e-3: khi tối ưu hóa còn chậm, việc giảm xung đột gradient giữa các thành phần loss giúp classifier học nhanh hơn rõ rệt.

Mỗi tập kiểm thử của target chỉ có khoảng 60 mẫu, nên một mẫu dự đoán sai tương ứng với khoảng 1.5 đến 1.7 điểm phần trăm Accuracy. Các chênh lệch dưới khoảng 2 điểm giữa các biến thể nên được xem là chưa có ý nghĩa thống kê. Ngoài ra kết quả trong tài liệu này dựa trên một lần chạy với seed 0; để có kết luận chắc chắn hơn cần chạy ba seed như thiết lập gốc của CoTMix và báo cáo cả độ lệch chuẩn.

Cuối cùng, DINOv3 ở đây vẫn được giữ frozen và tín hiệu cảm biến phải được chuyển sang ảnh GAF, nên khả năng biểu diễn của embedding bị giới hạn bởi cách ánh xạ chín kênh cảm biến vào ba kênh màu. Kết quả của DINOv3 vì thế vẫn thấp hơn đáng kể so với 1D-CNN (Accuracy khoảng 83% trên cùng cách đánh giá), và phần tinh chỉnh loss chỉ nhằm cải thiện tốc độ học trong giai đoạn đầu chứ không thu hẹp được khoảng cách về chất lượng biểu diễn giữa hai encoder.

## 10. Vị trí mã nguồn và cách chạy lại

Toàn bộ mã của các thành phần loss mới nằm trong tệp CoTMix/models/dinov3_loss.py, gồm lớp DiversityLoss và lớp DINOv3LossAddons. Trọng số được khai báo trong mục dinov3_overrides của lớp HAR trong CoTMix/configs/hparams.py và chỉ được trainer áp dụng khi backbone là DINOv3ViTS16. Tệp algorithms.py chỉ gọi các thành phần này trong hàm update của CoTMix.

Lệnh dùng để tái tạo kết quả 1 epoch trong tài liệu, chạy từ thư mục gốc của dự án:

```bash
py -3.13 main.py --backbone DINOv3ViTS16 --data_path CoTMix/data/ --save_dir CoTMix/experiments_logs --run_description dinov3_newloss_e1 --num_runs 1 --epochs 1 --batch_size 16 --device cpu
```
