# LSTM-VAE For Anomaly Detection
Deteksi anomali pada data deret waktu (time-series) menggunakan arsitektur LSTM dan latent space VAE.
## Main Ideas
- LSTM sebagai decoder dan encoder, juga karena kemampuannya dalam menangkap data dependensi waktu seperti time-series.
- Memanfaaatkan cara kerja dari latent space pada VAE, sebagai backbone untuk mengetahui dan memaksa data menjadi distribusi Gaussian (non-anomaly).
## Additional
Pada penelitian ini menggunakan pendekatan supervised, dimana dataset diharapkan memiliki kolom label yang menandai anomali terjadi. Soon, akan dikembangkan untuk mengatasi unsupervised. 
### Loss
- MSE Loss
- KL Divergence

### Data
Data yang digunakan dalam penelitian/pemodelan ini bersumber dari NAB

## Results Record
|Run|LR|BS|Hidden|Epcohs|Win|Stride|Beta|Final Loss|Recon Loss|KLD Loss|
|-|-|-|-|-|-|-|-|-|-|-|
|01|1e-3|32|64|50|10|1|1.0|0.978874|0.978797|0.000077|


## Clone
``` bash
git clone https://github.com/wildanjr19/vae-lstm-ts-anomaly-detection.git
```


### References