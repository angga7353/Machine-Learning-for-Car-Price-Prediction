# Model Predict Harga Mobil

## Repository Outline
Pada repository ini file yang saya push ada 3 file, yaitu :

├── deployment/
│   ├── app.py
│   └── eda.py
│   └── prediction.py
│   └── PipeLine_Regressi.pkl
├── description.md
├── P1M2_Angga.ipynb
├── P1M2_Angga_inf.ipynb
├── car_price_prediction_.csv
└── README.md

## Problem Background
`Spesific`    : Mengembangkan model prediksi harga mobil dengan tujuan untuk mengetahui faktor - faktor apa saja yang mempengaruhi harga mobil dipasar AS.

`Measurable`  : Model prediksi harga yang akan dibuat akan di evaluasi dengan menggunakan matrik `Mean Squared Error (MSE)`, `Mean Absolute Error` atau R-squared. Model akan dianggap berhasil atau akurat jika MSE atau MAE nya kurang dari `5%` dan R-squared nya sekitar `0.070`.

`Achievable`  : Model yang akan digunakan untuk prediksi harga menggunakan algoritma seperti `Decision Tree`, `Random Forest`, `KNN`, `SVM`, dan `Boosting`. Dari kelima algoritma ini akan di implementasikan dan akan dipilih salah satu algoritma yang terbaik untuk melakukan predict.

`Relevant`    : Tujuan ini relevan karena memberikan wawasan penting bagi perusahaan dalam memahami faktor - faktor yang mempengaruhi harga mobil di pasar AS. Dalam hal ini juga akan mendukung keputusan yang akan diambil oleh perusahaan untuk menentukan harga yang dapat bersaing dengan perusahaan - perusahaan besar di eropa.

`Time-Bounnd` : Membangun dan mengimplementasikan mobel prediksi dalam kurun waktu 3 bulan kedepan.


**Ringkasan Problem Statement**
Sebuah perusahaan mobil ingin mengembangkan model prediksi. Untuk model prediksi yang dibuat akan dilakukan evaluasi dngan `MSE`, `MAE`, dan `R-Squared`, Model akan dianggap berhasil jika MSE atau MAE nya kurang dari 5% dan R-Squared nya `0.070`. Lalu untuk algoritma model yang akan digunakan degan membandingkan model yang paling bagus dari algoritma `Decision Tree`, `Random Forest`, `KNN`, `SVM`, dan `Boosting`. Tujuan model prediksi ini dibuat untuk mengetahui harga kendaraan yang di jual oleh perusahaan - perusahaan besar di eropa. Lalu untuk membangun sebuah model prediksi ini membutuhkan waktu sekitar 3 bulan kedepan.

## Project Output
Pada bab ini saya akan memberikan kesimpulan akhir dari apa yang telah saya lakukan. Dari keseluruhan model yang telah saya coba sebelum melakukan `hyperparameter tunning` untuk keseluruhan model sebenarnya dapat melakukan predict harga modil akan tetapi jika dilihat dari `MAE`, `MSE`, dan `R2 Score`, yang paling bagus pada model `GradientBoostingRegressor` akan tetapi jika kita melihat dari hasil `cross-validation` sebenarnya yang terbaik adalah pada model `SVR`. Alasan saya menggunakan model `GradientBoostingRegressor` adalah karena tolak ukur yang dilihat yaitu menggunakan matrix `MAE`, `MSE` dan `R2-Score`. 

Setelah saya melakukan pemilihan model saya melakukan `hyperparameter tunning` pada model yang saya pilih dan menghasilkan hasil predict yang lebih baik dan hasil predict menggunakan model `GradientBoostingRegressor` yang telah dilakukan `hyperparameter tunning` dapat dikatakan berhasil atau sucses.

## Data
Data yang saya gunakan untuk melakukan analisys ini adalah data yang bersumber dari kaggle, pada dataset ini terdapat jumlah column sebanyak 10 column, lalu untuk baris data sebanyak 2.500, untuk data yang saya gunakan ini adalah data yang telah bersih dari missing values dan tidak ada outlier.

## Method
Pada project ini saya menggunakan beberapa model untuk melakukan predict ini seperti, berikut :
1. KNN
2. SVR
3. Decision Tree
4. Random Forest
5. Boosting

Dari 5 model yang telah saya kerjakan saya menggunakan model Boosting karena model ini merupakan model terbaik untuk melakukan predict harga pada case saya.


## Stacks
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import joblib
import streamlit as st

## Reference
Link DataSet    : [DataSet](https://www.kaggle.com/datasets/zafarali27/car-price-prediction/data)

Link Hugging Face   : [Hugging Face](https://huggingface.co/spaces/angga770773/Project_Predict_Price_Car)

Link Acuan Measurable : [Link 1](https://www.sciencedirect.com/science/article/pii/S0957417424025077), [Link 2](https://www.researchgate.net/publication/386117316_Predicting_Vehicle_Prices_Using_Machine_Learning_A_Case_Study_with_Linear_Regression)

link Acuan Feature Selection : [link_1](https://www.suara.com/otomotif/2023/12/11/210056/ini8-faktor-yang-mempengaruhi-harga-jual-mobil-makin-standar-makin-oke?page=all),
[link_2](https://rustpro.id/faktor-yang-mempengaruhi-nilai-jual-mobil/)
