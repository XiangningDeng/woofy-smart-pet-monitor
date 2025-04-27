import torch
import torch.nn as nn

class DogBehaviorTransformerFlash(nn.Module):
    def __init__(self, input_channels=12, seq_len=201, num_classes=12,
                 d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # 更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model + input_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)              # [B, T, C]
        z = self.input_proj(x)              # [B, T, d_model]
        z = self.encoder(z)                 # [B, T, d_model]
        z = z.mean(dim=1)                   # [B, d_model]
        fft_part = x[:, -1, :]              # [B, C] FFT
        z = torch.cat([z, fft_part], dim=1) # [B, d_model + C]
        return self.classifier(z)
