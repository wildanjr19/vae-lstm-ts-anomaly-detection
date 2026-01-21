# LSTM-VAE For Anomaly Detection
Deteksi anomali pada data deret waktu (time-series) menggunakan arsitektur LSTM dan latent space VAE.

## Main Ideas
- LSTM sebagai decoder dan encoder. LSTM sangat membantu dalam menangkap pola long-term yang memiliki dependensi pada waktu atau urutan. 
- VAE Latent Space menerapkan distribusi normal pada data non-anomali menggunakan KL-Divergence, sehingga anomali lebih mudah dideteksi.

## Loss
Kombinasi antara Mean Squared Eror (MSE) untuk evaluasi rekonstruksi dan Kullback-Leibler (KL) Divergence untuk regularisasi.
- **MSE** : Evaluasi antara data real dan hasil rekonstruksi  
  $$\text{MSE} = \frac{1}{N \times D} \sum_{i=1}^{N} \sum_{j=1}^{D} (x_{ij} - \hat{x}_{ij})^2$$
- **KL Divergence** : Mengukur perbedaan antara dua distribusi  
  $$D_{KL}(q(z|x) | p(z)) = -\frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$$

## Architecture
Arsitektur model:
- Encoder : Lapisan LSTM akan memetakan input urutan data deret waktu ke nilai rata-rata (mu) dan varians (sigma/logvar) di laten space.
- Sampling Layer : Mengambil sampel dari distribusi di latent space dengan menggunakan reparameterization trick.
- Decoder : Lapisan LSTM merekonstruksi urutan asli dari sampling latent space.
  
## Data
Dataset yang digunakan dalam proyek ini bersumber dari [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB), sebuah benchmark standar untuk deteksi anomali deret waktu. NAB mencakup beragam dataset dunia nyata dan buatan dengan anomali yang diberi label.

## Results Record
### Training
|Run|LR|BS|Hidden|Epcohs|Win|Stride|Beta|Final Loss|Recon Loss|KLD Loss|
|-|-|-|-|-|-|-|-|-|-|-|
|01|1e-3|32|64|50|10|1|1.0|0.978874|0.978797|0.000077|
### Test

## Usage
### Clone
``` bash
git clone https://github.com/wildanjr19/vae-lstm-ts-anomaly-detection.git
```
### Running
Pastikan data (raw) ada di `data/`
``` bash
python src/train.py --dataset data/{data.csv} --epochs 50 --batch_size 32
```

## Future Work
- [ ] Bisa handle unuspervised.
- [ ] LSTM + Attention.

## References
- [Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model](https://ieeexplore.ieee.org/document/9053558)
- [Anomaly Detection Using LSTM-Based Variational Autoencoder in Unsupervised Data in Power GridDibyajy](https://www.ece.nus.edu.sg/stfpage/bsikdar/papers/sj_dg_23.pdf)
- [TS VAE-LSTM](https://github.com/thatgeeman/ts_vae-lstm)

## My Notes
- `contaminaton` mencegah overfitting karena pada kasus real data pelatihan bisa saja mengandung anomali atau pengumpulan data yang tidak sepenuhnya sempurna.