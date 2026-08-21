# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Lê Văn Tuệ |
| MSSV | 2A202601048 |
| Lớp / Khóa | K4 |
| Repo GitHub | [github.com/LeTue3004/Track2_Day21_2A202601048_LeVanTue](https://github.com/LeTue3004/Track2_Day21_2A202601048_LeVanTue) |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 2 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ thứ ba có F1-score cao nhất và vượt ngưỡng triển khai 0.65. Accuracy cao nhất thuộc về bộ thứ hai, không trùng với F1 cao nhất; điều này cho thấy accuracy không phản ánh đầy đủ lớp thu nhập cao là lớp thiểu số. Với Gradient Boosting, learning rate thấp thường cần nhiều cây hơn.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tỷ lệ lớp thu nhập cao trong dữ liệu Adult chỉ khoảng 24,8%, nên đây là bài toán mất cân bằng lớp. Mô hình luôn dự đoán `thu_nhap_thap` vẫn có accuracy xấp xỉ 0,752, nhưng F1 lớp dương bằng 0. Vì vậy quality gate dùng F1-score của target bằng 1: chỉ số này kết hợp precision và recall, phản ánh khả năng nhận diện và không bỏ sót người thu nhập cao. Accuracy chỉ được log để tham khảo. Tôi dùng `f1_score(y_eval, preds)` mặc định cho lớp dương, không dùng `average="weighted"` hoặc `average="macro"` vì các cách này có thể che giấu hiệu năng kém của lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| DVC pull trên GitHub Actions bị 403 | IAM policy thiếu `s3:GetObject` | Bổ sung quyền đọc object cho đúng bucket S3. |
| Unit test CI trả về `NoneType` | Phiên bản test hoàn thiện ở local chưa được push | Commit và push `src/train.py` cùng `tests/test_train.py`. |
| EC2 chưa phục vụ API | Model chỉ xuất hiện sau khi pipeline Train và Release thành công | Cấu hình systemd, IAM role EC2 và kiểm tra `/healthz` sau release. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---:|---:|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Thêm 22.361 mẫu làm F1 tăng 0.0205 và accuracy tăng 0.0080. Commit dữ liệu kích hoạt DVC, train, quality gate và deploy.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 2 - Threshold 0.30 đạt F1 0.7537, cao hơn F1 0.7354 tại threshold mặc định 0.50.
- [x] Bonus 3 - `outputs/detail.txt` tự ghi confusion matrix, precision và recall; lớp thu nhập cao có precision 0.7014, recall 0.8145.
- [x] Bonus 5 - Tỷ lệ lớp dương 24,78%, lệch 0,02 điểm phần trăm so với mốc 24,8%; pipeline cảnh báo khi lệch quá 5 điểm phần trăm.
