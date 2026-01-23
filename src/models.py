import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_LSTM(nn.Module):
    """
    Model VAE-LSTM untuk deteksi anomali pada data time series.
        - LSTM berperan sebagai encoder dan decoder.
        - VAE digunakan untuk memodelkan distribusi di latent space. Dari sini, VAE akan memahami karakteristik dari data yang normal.
    Args:
        - input_dim (int) : Dimensi input.
        - hidden_dim (int) : Dimensi hidden state LSTM.
        - latent_dim (int) : Dimensi di latent space untuk VAE.
        - lstm_layers (int) : Banyak layer LSTM.
        - dropout (float) : Dropout rate pada LSTM.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 16, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lstm_layers = lstm_layers
        
        # ENCODER
        self.lstm_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Encoder -> Latent
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Latent -> Decoder
        self.latent_to_decoder = nn.Linear(latent_dim, hidden_dim)

        # DECODER
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Decoder -> Output
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encoder: x -> mu dan logvar.
        Disini kita akan mendapatkan representasi latent dari input x, yaitu mu dan logvar.
        Kita ambil last hidden state dari LSTM, kemudian dimasukkan ke layer hidden_to_mu dan hidden_to_logvar.
        """
        lstm_out, (hidden, cell) = self.lstm_encoder(x)
        # ambil last hidden
        last_hidden = hidden[-1]

        # get mu dan logvar
        mu = self.hidden_to_mu(last_hidden)
        logvar = self.hidden_to_logvar(last_hidden)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick untuk sampling z.
        Dilakukan agar gradient dapat mengalir di sampling, atau mengubah operasi stokastik (randomness) menjadi operasi deterministik.
        z = mu + eps * std
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        """
        Decoder: z -> reconstructed x.
        Hasil dari reparameterization (z) dimasukkan ke layer laten_to_decoder atau hidden.
        Kemudian di-expand untuk menghindari missmatch : [bh, latent_dim] -> [num_layers, bh, hidden_dim]
        """
        hidden = self.latent_to_decoder(z)
        # expand latent untuk setiap time step -> [num_layers, bh, hidden_dim] 
        # dilakukan untuk initial hidden state pada LSTM decoder, 
        # alternatif jika input sequence ke lstm [bh, seq_len, hidden_dim]
        hidden = hidden.unsqueeze(0).repeat(self.lstm_layers, 1, 1)  # initial hidden state : [num_layers, bh, hidden_dim]
        cell = torch.zeros_like(hidden) # initial cell state likes hidden

        # LSTM sudah punya memori dari intial hidden state sebelumnya
        # jadi decodor input 0
        decoder_input = torch.zeros(z.size(0), seq_len, self.hidden_dim).to(z.device)  # [bh, seq_len, hidden_dim]

        # decoder
        lstm_out, _ = self.lstm_decoder(decoder_input, (hidden, cell)) # lstm_out: [bh, seq_len, hidden_dim]

        # decoder -> output
        output = self.hidden_to_output(lstm_out)  # [bh, seq_len, input_dim]

        return output
    
    def forward(self, x):
        """
        Forward pass
            - get mu dan logvar dari encode
            - get sampling z dari reparameterization
            - get reconstructed (output) dari hidden ke decode
        seq_len = x.size(1)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z, x.size(1))  

        return x_reconstructed, mu, logvar
    
    def reconstruct(self, x):
        """
        Fungsi untuk rekosnstruksi input x tanpa sampling pakai mu sebagai z.
        Untuk evaluasi.
        """
        mu, logvar = self.encode(x)
        x_reconstructed = self.decode(mu, x.size(1))
        return x_reconstructed
