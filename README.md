# LSTM-VAE For Anomaly Detection
Deteksi anomali pada data deret waktu (time-series) menggunakan arsitektur LSTM dan latent space VAE.
## Main Ideas
- LSTM sebagai decoder dan encoder, juga karena kemampuannya dalam menangkap data dependensi waktu seperti time-series.
- Memanfaaatkan cara kerja dari latent space pada VAE, sebagai backbone untuk mengetahui dan memaksa data menjadi distribusi Gaussian (non-anomaly).
## Additional
Pada penelitian ini menggunakan pendekatan supervised, dimana dataset diharapkan memiliki kolom label yang menandai anomali terjadi. Soon, akan dikembangkan untuk mengatasi unsupervised. 
### Clone
### Data
Data yang digunakan dalam penelitian/pemodelan ini bersumber dari NAB
### References