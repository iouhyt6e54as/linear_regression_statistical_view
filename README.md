# 📐 Linear Regression — Statistical View

> A from-scratch implementation of Simple Linear Regression with full statistical inference: F-statistic, confidence intervals, R², and predictor comparison — built in pure NumPy/SciPy with no sklearn.

---

## 📌 Project Overview

This project implements Simple Linear Regression from a **statistical inference perspective**, going beyond just fitting a line. It computes the full ANOVA decomposition, hypothesis testing, and confidence intervals — the way a statistician would analyze a regression model.

The module is used to predict **house prices** from the King County dataset and compare the predictive power of two features (`sqft_living` vs `bedrooms`) using F-statistics.

---

## 📁 Project Structure

```
project/
│
├── linear_regression_statistical_view.py   # Core model class + compare() utility
├── project_2.ipynb                         # Usage notebook with KC house data
├── cleaned_kc_house_data.csv               # Preprocessed dataset
└── README.md
```

---

## 🔬 What the Model Computes

After calling `.fit(X, Y)`, the model produces the full statistical breakdown:

| Quantity | Description |
|---|---|
| `β₀`, `β₁` | Intercept and slope (OLS closed-form) |
| `SST` | Total Sum of Squares |
| `SSR` | Regression Sum of Squares |
| `SSE` | Error Sum of Squares |
| `MSR`, `MSE` | Mean squares for model and error |
| `F-statistic` | Model significance test: MSR / MSE |
| `F-critical` | Threshold at given α (default 0.05) |
| `R²` | Coefficient of determination |
| `R` | Pearson correlation (signed) |
| `df_model`, `df_resid`, `df_total` | Degrees of freedom |
| CI for β₀, β₁ | t-based confidence intervals at level α |

---

## 🧮 Key Formulas

```
β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
β₀ = ȳ − β₁ × x̄

SST = Σ(yᵢ - ȳ)²
SSR = Σ(ŷᵢ - ȳ)²
SSE = Σ(yᵢ - ŷᵢ)²

F = MSR / MSE = (SSR / 1) / (SSE / (n-2))
R² = SSR / SST
```

---

## 🗂️ API Reference

### `LinearRegressionStatisticalView(alpha=0.05)`

```python
model = LinearRegressionStatisticalView(alpha=0.05)
model.fit(X, Y)
```

| Method | Returns | Description |
|---|---|---|
| `.fit(X, Y)` | `self` | Fits OLS and computes all statistics |
| `.f_statistic()` | `float` | Computed F-value |
| `.f_critical()` | `float` | F-critical at given α |
| `.r_squared()` | `float` | R² score |
| `.r()` | `float` | Pearson correlation coefficient |
| `.beta_confidence_intervals()` | `dict` | CIs for β₀ and β₁ |
| `.degrees_of_freedom()` | `dict` | df_model, df_resid, df_total |
| `.summary()` | `dict` | All of the above in one call |

---

### `compare(df, x_col_1, x_col_2, y_col, alpha=0.05)`

Fits two separate models and selects the better predictor based on F-statistic.

```python
result = compare(df, "sqft_living", "bedrooms", "price")
# → {"best_predictor": "sqft_living", "f_stat_1": 16912.57, "f_stat_2": 1966.44, ...}
```

---

## 📊 Results on KC House Data

**Model:** `sqft_living` → `price`

| Metric | Value |
|---|---|
| β₀ (Intercept) | ~537,255 |
| β₁ (Slope) | ~248,853 |
| F-statistic | 16,912.58 |
| F-critical (α=0.05) | 3.84 |
| R² | 0.4945 |
| R (Pearson) | 0.7032 |
| df_model / df_resid | 1 / 17,288 |

F-stat (16,912) >> F-critical (3.84) → **the model is highly significant**.

**Predictor Comparison:**

| Predictor | F-statistic |
|---|---|
| `sqft_living` | 16,912.58 ✅ |
| `bedrooms` | 1,966.44 |

`sqft_living` is the stronger predictor of house price.

---

## 🛠️ Dependencies

```bash
pip install numpy scipy pandas
```

| Library | Purpose |
|---|---|
| `numpy` | All matrix and statistical computations |
| `scipy.stats` | F and t critical values |
| `pandas` | Data loading in the notebook |

---


## 👩‍💼 Author

**Shahd Ahmed Farghaly**
*Data Science Student — Alexandria University*

📧 [shahdfarghaly2005@gmail.com](mailto:shahdfarghaly2005@gmail.com)
🔗 [LinkedIn Profile](https://www.linkedin.com/in/shahd-farghaly-bb9356332)
