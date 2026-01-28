# Deep Learning – Practice Exercises Summary

Repo này tổng hợp toàn bộ các bài thực hành trong môn **Deep Learning**, bao gồm phân loại ảnh sử dụng mạng nơ-ron, sử dụng pre-trained mode & làm quen với Tensorboard, sử dụng thư viện transformers.
Mỗi bài được lưu trong thư mục **notebooks/** (ngoại trừ Practice 3).

---

## 📁 Cấu trúc thư mục

```
DeepLearning_Groups_8/
│
├── HuynhHau_CamGiang_Practices_3/
│   ├── configs/
│   ├── notebooks/
│   ├── src/
│   ├── outputs/
│   └── requirements.txt
│
├── notebooks/
│   ├── Practice1_PyTorch FashionMNIST Classification.ipynb
│   └── Practice2_Pre-trained model CNNs and Transfer Learning.ipynb
│
├── doc/
│   └── DeepLearning_Practice_Report_Group8.pdf
│
├── requirements.txt
└── README.md

```

---

# 📌 Practice 1 – PyTorch FashionMNIST Classification

**Người thực hiện:** _Phạm Gia Bảo – Dương Hưng_
**Notebook:** `Pracice1_PyTorch FashionMNIST Classification.ipynb`

**Tóm tắt nội dung:**

- Chuẩn bị dữ liệu FashionMNIST, áp dụng augmentation và chuẩn hóa.
- Xây dựng mô hình CNN nhiều lớp và huấn luyện bằng PyTorch.
- Theo dõi loss/accuracy, đánh giá mô hình bằng Accuracy.
- Lưu lại mô hình tốt nhất trong quá trình huấn luyện.

---

# 📌 Practice 2 – Pre-trained model CNNs & Transfer Learnings And Tensorboard

**Người thực hiện:** _Lê Văn An_
**Notebook:** `Practice2_Pre-trained model CNNs and Transfer Learnings.ipynb`

**Tóm tắt nội dung:**

- Sử dụng CNNs pretrained, thay hoặc fine-tune lớp phân loại cuối. Dataset: CIFAR10
- Huấn luyện mô hình trên dataset đã chuẩn bị.
- Đánh giá chi tiết hơn bằng Precision, Recall, F1-score.
- So sánh loss và acc của các model pre-trained, lưu lại và hiển thị trên Tensorboard

---

# 📌 Practice 3 – Get started with Hugging Face

**Người thực hiện:** _Huỳnh Hậu - Huỳnh Cẩm Giang_
**Folder:** `HuynhHau_CamGinag_Practice_3`

**Tóm tắt nội dung:**

- Dùng pretrained model Hugging Face để làm Sentiment Analysis
- Tokenize
- Thực hiện inference và lưu kết quả.
- Tiền xử lý dữ liệu cho bài toán **Binary Text Classification** (clean text, label → 0/1, xử lý trùng/xung đột).
- Fine-tune pretrained model bằng `Trainer`, đánh giá bằng **Accuracy** và **F1**

---

## 🎯 Mục đích của repo

- Tổng hợp toàn bộ bài thực hành Deep Learning trong môn học.
- Hệ thống hoá quy trình chuẩn của các bài toán Deep learning.
