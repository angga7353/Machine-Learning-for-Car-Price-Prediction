# Machine Learning for Car Price Prediction
## DataSet and Deployment
[DataSet](https://www.kaggle.com/datasets/zafarali27/car-price-prediction/data)
[Hugging Face](https://huggingface.co/spaces/angga770773/Project_Predict_Price_Car)

# Description Project
Sebuah perusahaan mobil berencana mengembangkan sebuah model prediksi harga kendaraan untuk mengetahui estimasi harga jual kendaraan yang dipasarkan oleh perusahaan-perusahaan besar di Eropa.
Model prediksi ini akan dievaluasi menggunakan metrik berikut:
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R-Squared (R²)
Kriteria keberhasilan model adalah apabila nilai MSE atau MAE kurang dari 5%, dan nilai R-Squared minimal 0.070.
Dalam pengembangan model, akan dilakukan perbandingan performa antara beberapa algoritma machine learning populer, yaitu:
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Boosting
Proses pembangunan model prediksi ini diperkirakan membutuhkan waktu sekitar 3 bulan ke depan.
# Conclusion
kesimpulan akhir dari seluruh proses pengembangan model prediksi harga mobil yang telah dilakukan.
Sebelum dilakukan hyperparameter tuning, semua model yang dicoba mampu melakukan prediksi harga mobil. Namun, berdasarkan evaluasi menggunakan metrik MAE, MSE, dan R² Score, model dengan performa terbaik secara metrik adalah GradientBoostingRegressor. Meski demikian, hasil cross-validation menunjukkan bahwa model Support Vector Regression (SVR) memiliki performa terbaik secara generalisasi.
Alasan utama saya memilih model GradientBoostingRegressor adalah karena fokus pada tolok ukur utama, yakni metrik MAE, MSE, dan R² Score.
Setelah pemilihan model, saya melakukan proses hyperparameter tuning pada model tersebut, yang menghasilkan peningkatan performa prediksi secara signifikan. Dengan demikian, model GradientBoostingRegressor yang telah di-tuning dapat dikatakan berhasil dan sukses dalam memprediksi harga mobil dengan akurasi yang baik
# Data Source and Description
Data yang digunakan untuk analisis ini bersumber dari Kaggle, sebuah platform dataset publik yang terpercaya.
* Dataset terdiri dari 10 kolom yang memuat berbagai atribut terkait mobil.
* Jumlah baris data dalam dataset adalah 2.500 entri.
* Data telah melalui proses pembersihan sehingga bebas dari missing values dan tidak terdapat outlier, sehingga siap untuk dianalisis dengan hasil yang valid dan akurat.
# Models Used
Dalam proyek ini, saya menguji beberapa model machine learning untuk melakukan prediksi harga mobil, yaitu:
* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* Boosting
Dari kelima model tersebut, model Boosting dipilih sebagai model terbaik karena memberikan hasil prediksi harga yang paling akurat dan konsisten untuk kasus yang saya tangani.
# Libraries and Tools Used
Dalam proyek ini, beberapa library dan tools utama Python digunakan untuk analisis data, pemodelan, dan deployment aplikasi, antara lain:
* pandas: Manipulasi dan pengolahan data.
* matplotlib.pyplot dan seaborn: Visualisasi data statistik dan grafik yang informatif.
* numpy: Operasi numerik dan manipulasi array.
* scipy.stats (pearsonr, chi2_contingency): Uji korelasi dan uji independensi statistik.
* scikit-learn (sklearn):
    * train_test_split, cross_val_score, KFold: Membagi data dan evaluasi model dengan cross-validation.
    * OneHotEncoder, StandardScaler: Preprocessing data kategorikal dan numerik.
    * DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, KNeighborsRegressor, SVR: Berbagai algoritma regresi untuk prediksi harga mobil.
    * mean_absolute_error, mean_squared_error, r2_score: Metode evaluasi performa model.
    * Pipeline, make_pipeline, ColumnTransformer: Membangun pipeline pemrosesan dan modeling yang modular dan efisien.
    * PCA: Reduksi dimensi fitur.
* joblib: Menyimpan dan memuat model yang sudah dilatih (model persistence).
* streamlit: Membangun aplikasi web interaktif untuk demo model prediksi harga mobil.
Penggunaan library tersebut memastikan workflow analisis dan pengembangan model berjalan lancar dari preprocessing data, pelatihan, evaluasi, hingga penyajian hasil secara interaktif.
