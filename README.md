# 🏨 Hotel Booking Demand — Exploratory Data Analysis

An end-to-end EDA of ~119k hotel reservations (City Hotel & Resort Hotel, Portugal, 2015–2017): data cleaning, leakage detection, outlier handling, visual analysis and an ML-readiness check.

> AIML EDA Minor Project · RV College of Engineering · Mentor: Sruthi Tarimana

## 📌 Objective
1. Audit and fix the data quality of the raw file.
2. Understand who books, when and how (guest profile, channels, seasonality).
3. Find the factors most associated with **cancellation** (`is_canceled`) and **price** (`adr`).
4. Produce a clean dataset (`df_clean`) and judge its readiness for machine learning.

## 📂 Dataset
- **Source:** [Hotel Booking Demand on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Original paper:** Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22
- **Size:** 119,390 rows × 32 columns (raw) → 87,212 rows × 37 columns (cleaned)

## 🔧 What was done
| Step | Result |
|---|---|
| Duplicates | 31,994 exact duplicates (26.8%) removed; cancellation rate moved from 37.0% to 27.5% |
| Missing values | `company`/`agent` → flags (`has_company`, `has_agent`); `country` → "Unknown"; `children` → 0 |
| Cleaning | Merged `CN`/`CHN`, handled `Undefined` placeholders, fixed data types, removed 167 impossible bookings |
| Data leakage | `reservation_status` and `reservation_status_date` proven to leak the target and dropped |
| Outliers | Judged per variable using business logic; only 17 rows removed (e.g. ADR = 5,400, 55 adults) |
| Analysis | Categorical, numerical, univariate, bivariate (Cramér's V, Spearman) and multivariate (heatmaps, VIF) |
| ML-readiness | Baseline Random Forest with a chronological split: ROC-AUC ≈ 0.83 |

## 🔍 Key findings
- Cancellation risk rises with **lead time** (8% within a week → 41% beyond a year) and is highest for **Online TA** bookings.
- Guests with special requests and repeat guests cancel far less (repeat guests: 7.7% vs 28.3%).
- Prices are strongly seasonal: Resort ADR swings from about €50 (Nov) to about €189 (Aug).
- Some features look predictive but are only recorded at check-in (room mismatch, parking), so they were excluded from the model.

## 🚀 How to run
```bash
git clone https://github.com/<your-username>/hotel-booking-eda.git
cd hotel-booking-eda
pip install -r requirements.txt
# download hotel_bookings.csv from the Kaggle link above and place it in this folder
jupyter notebook Varun_HotelBookings_EDA_Project.ipynb
```
Run all cells from top to bottom. The notebook writes `hotel_bookings_cleaned.csv`.

## 🗂️ Repository structure
```
├── Varun_HotelBookings_EDA_Project.ipynb   # full analysis
├── hotel_bookings_cleaned.csv              # final cleaned dataset (df_clean)
├── requirements.txt
└── README.md
```

## 🛠️ Tech stack
Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · scikit-learn

## 👤 Author
**Varun** — BE Computer Science, RV College of Engineering, Bengaluru
