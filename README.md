# Heart Disease Prediction (Cleveland)

Dự án dự đoán bệnh tim sử dụng bộ dữ liệu Cleveland. Kho lưu trữ được tổ chức xoay quanh 3 notebook chính: tiền xử lý, huấn luyện mô hình, và giải thích mô hình. README này tập trung vào cách chạy, cấu trúc dự án, và tóm tắt nội dung đã thực hiện trong các notebook.

Lưu ý: README không đề cập phần giao diện; bạn có thể bổ sung riêng khi cần.


## Môi trường và cài đặt

- Yêu cầu Python 3.9+ (khuyến nghị 3.10).
- Cài đặt phụ thuộc (đã có `requirements.txt`):

```
# Tạo và kích hoạt môi trường (tuỳ chọn)
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# Cài gói
pip install -r requirements.txt

# Tuỳ chọn (nếu dùng):
# pip install xgboost seaborn matplotlib ipywidgets
```

Gợi ý: nếu dùng Conda/Anaconda, có thể `conda create -n ml python=3.10 && conda activate ml` rồi `pip install -r requirements.txt`.


## Cấu trúc thư mục

```
.
├── notebooks/
│   ├── 1. Preprocessing.ipynb      # Tiền xử lý, tạo các biến thể dữ liệu và tách train/val/test
│   ├── 2. Modeling.ipynb           # Huấn luyện, tối ưu tham số, so sánh mô hình
│   └── 3. Explaination.ipynb       # Giải thích mô hình (SHAP)
├── splits/                         # Thư mục sinh ra sau Preprocessing (các CSV đã tách)
├── cleveland.csv                   # Dữ liệu gốc (nếu sử dụng)
├── requirements.txt                # Danh sách thư viện
└── (các tệp khác)
```

Thư mục `splits/` được tạo tự động bởi notebook tiền xử lý, chứa các tập dữ liệu đã sẵn sàng để huấn luyện.


## Quy trình chạy

1) Preprocessing (notebooks/1. Preprocessing.ipynb)
- Mục tiêu: Đọc dữ liệu gốc, làm sạch/tiền xử lý, tạo đặc trưng, chọn đặc trưng và tách train/val/test. Kết quả được ghi xuống `splits/`.
- Các điểm chính đã làm:
  - Thiết lập hạt giống ngẫu nhiên cố định (42) để tái lập.
  - Xử lý cột mục tiêu (`target`) và chuẩn hoá kiểu dữ liệu.
  - Pipeline tiền xử lý (impute, scale, one-hot encode) cho cả đặc trưng số và phân loại; hỗ trợ xuất dạng pandas.
  - Thêm đặc trưng mới (ví dụ: tỷ lệ/biến tương tác đơn giản) qua `AddNewFeaturesTransformer`.
  - Chọn đặc trưng theo Mutual Information (MI) trên không gian đặc trưng đã biến đổi để lấy Top‑K; tạo các biến thể dữ liệu:
    - `raw_*`: bộ đặc trưng gốc sau tiền xử lý cơ bản.
    - `dt_*`: chọn Top‑K đặc trưng từ cây quyết định (feature_importances_).
    - `fe_*`: đặc trưng đã kỹ thuật hoá + one‑hot; chọn Top‑K theo MI.
    - `fe_dt_*`: rút gọn thêm từ `fe_*` bằng quan trọng hoá dựa trên cây.
  - Lưu các tệp CSV: `raw_train/val/test.csv`, `dt_train/val/test.csv`, `fe_train/val/test.csv`, `fe_dt_train/val/test.csv`.
  - Cân bằng lớp (tùy chọn): cell cuối dùng SMOTE để tạo `fe_train_smote.csv` (và nếu có, `fe_dt_train_smote.csv`). Validation/test được giữ nguyên để đánh giá trung thực.

2) Modeling (notebooks/2. Modeling.ipynb)
- Mục tiêu: Huấn luyện và tối ưu tham số nhiều thuật toán trên các biến thể dữ liệu, chọn mô hình tốt theo ROC‑AUC trên validation, và đánh giá trên test.
- Các điểm chính đã làm:
  - Các thuật toán: RandomForest, AdaBoost, GradientBoosting, LightGBM, XGBoost, CatBoost (cài thêm nếu cần).
  - Không gian tham số được đề xuất trong `SUGGESTERS`, tối ưu bằng Optuna (`optuna_tune_simple`) và có biến thể hỗ trợ early stopping (`optuna_tune_es`) cho XGB/LGBM/CAT.
  - Chạy theo từng biến thể dữ liệu: `raw`, `dt`, `fe`, `fe_dt`, và (nếu có) `fe_smote` (train SMOTE + val/test gốc).
  - Ghi kết quả: bảng tổng hợp theo dataset/mô hình (ví dụ lưu `splits/modeling_summary.csv`) và có cell in ra `best_estimator` rồi lưu chung một tệp CSV tổng hợp.
  - Biểu đồ: có các cell dựng 6 biểu đồ (theo 6 thuật toán), hiển thị điểm của từng dataset (val và test) để so sánh trực quan.
  - Lưu ý tương thích:
    - sklearn ≥ 1.7: AdaBoost dùng `estimator=DecisionTreeClassifier(...)` (không dùng `base_estimator`).
    - XGBoost: tuỳ phiên bản, early stopping dùng `callbacks=EarlyStopping(...)` hoặc `early_stopping_rounds`; notebook đã hỗ trợ tự phát hiện và rẽ nhánh phù hợp.
    - CatBoost: nếu gặp lỗi ghi thư mục tạm, cấu hình `train_dir` tới thư mục có quyền ghi.

3) Explaination (notebooks/3. Explaination.ipynb)
- Mục tiêu: Giải thích mô hình bằng SHAP (tổng quan và theo từng mẫu).
- Các điểm chính đã làm:
  - Nạp mô hình tốt nhất đã huấn luyện (khuyến nghị lưu/đọc `.joblib`/`.pkl` từ notebook Modeling) hoặc fit một mô hình ví dụ rồi giải thích.
  - Dùng `shap.TreeExplainer` cho mô hình cây (RF, AdaBoost, XGB, LGBM, CatBoost). Fallback sang `KernelExplainer` nếu cần.
  - Biểu đồ: `summary_plot` (beeswarm và bar) và `waterfall` cho một mẫu cụ thể.
  - Lỗi thường gặp: "'RandomForestClassifier' object has no attribute 'estimators_'" là do mô hình chưa `.fit()`; hãy fit hoặc load mô hình đã huấn luyện trước khi tính SHAP.
  - Đảm bảo cột của dữ liệu giải thích khớp với cột khi mô hình được huấn luyện (đặc biệt với one‑hot/FE).


## Tái lập thí nghiệm nhanh

1. Mở 1. Preprocessing.ipynb và chạy toàn bộ để tạo thư mục `splits/` và các biến thể dữ liệu.
2. Mở 2. Modeling.ipynb và chạy nhóm cell tương ứng biến thể bạn muốn đánh giá. Nếu cần, dùng hàm hỗ trợ early stopping khi gọi các model có hỗ trợ.
3. Mở 3. Explaination.ipynb, nạp mô hình tốt nhất (hoặc fit lại), và chạy các cell SHAP để giải thích.

Tất cả notebook đều đặt seed 42 để các kết quả có thể tái lập ở mức hợp lý.


## Mẹo & xử lý sự cố

- Thiếu tiến trình trong Jupyter khi dùng một số thư viện (tqdm/lazypredict): cài `ipywidgets` và bật nbextension nếu cần.
- CatBoost trên Windows/OneDrive: chỉ định `train_dir` tới thư mục có quyền ghi.
- SHAP chậm với KernelExplainer: hãy `sample` 100–200 dòng để hiển thị nhanh.


## Dữ liệu đầu ra khi thí nghiệm

- `splits/raw_*.csv`, `splits/dt_*.csv`, `splits/fe_*.csv`, `splits/fe_dt_*.csv` — các biến thể dữ liệu đã tách train/val/test.
- `splits/fe_train_smote.csv` (tuỳ chọn) — tập train đã cân bằng bằng SMOTE.
- `splits/modeling_summary.csv` — tổng hợp hiệu quả theo dataset và mô hình (nếu đã chạy cell lưu).
- `splits/best_estimators_summary.csv` — thông tin `best_estimator` theo từng dataset/mô hình (nếu đã chạy cell lưu).

---


