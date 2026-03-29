#  Model Performance Report

##  Model Overview
Model yang digunakan adalah **Logistic Regression** yang dibangun menggunakan **scikit-learn Pipeline**.

Pipeline mencakup:
- ColumnTransformer
- OrdinalEncoder (categorical features)
- StandardScaler (numerical features)
- SimpleImputer (handling missing values)

---

##  Feature Engineering

Fitur tambahan yang digunakan:

### 🔹 Cabin-based
- Deck
- CabinNum
- Side

### 🔹 Passenger-based
- GroupSize
- IsSolo

### 🔹 Family-based
- FamilySize

### 🔹 Spending features
- TotalSpending
- HasSpending

### 🔹 Age features
- AgeGroup

### 🔹 Missing indicators
- AgeMissing
- CryoSleepMissing
- VIPMissing

---

##  Data Split

- Training set: 6954 samples
- Validation set: 1739 samples

---

## Cross Validation

Accuracy scores: 
[0.7829, 0.7901, 0.7901, 0.7894, 0.7949]


- Mean Accuracy: **0.7895**
- Std Dev: **0.0039**

Artinya Model cukup stabil (variance kecil)

---

##  Final Evaluation

### Accuracy
**0.7878**

---

### Classification Report

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| 0 | 0.79 | 0.78 | 0.78 |
| 1 | 0.79 | 0.80 | 0.79 |

---

### Confusion Matrix

- Mean Accuracy: **0.7895**
- Std Dev: **0.0039**

Artinya model cukup stabil (variance kecil)

---

## Final Evaluation

### Accuracy
**0.7878**

---

### Classification Report

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| 0 | 0.79 | 0.78 | 0.78 |
| 1 | 0.79 | 0.80 | 0.79 |

---

### Confusion Matrix
[[673 190]
[179 697]]


---

## Analysis

- Model memiliki performa yang **cukup seimbang** antara class 0 dan class 1
- Tidak ada indikasi overfitting karena:
  - CV score ≈ Validation score
- Recall class 1 sedikit lebih tinggi → model cukup baik mendeteksi penumpang yang transported

---

## Conclusion

Model Logistic Regression dengan Pipeline:
- Stabil dan konsisten
- Mudah diinterpretasikan
- Siap untuk deployment (Streamlit)

---

## Deployment

Model telah disimpan sebagai:
models/pipeline.pkl


Dan dapat digunakan langsung di aplikasi Streamlit.

---

## Future Improvements

- Tambahkan model lain (Random Forest, XGBoost)
- Gunakan OneHotEncoder untuk fitur nominal
- Hyperparameter tuning
- Feature selection lanjutan