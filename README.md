# Machine Learning for Car Price Prediction

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
