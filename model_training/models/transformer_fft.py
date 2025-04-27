import torch
import torch.nn as nn

class DogBehaviorTransformer(nn.Module):
    def __init__(self, input_channels=12, seq_len=201, num_classes=12,
                 d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # 加速度和陀螺仪分开处理，分配3通道给加速度，3通道给陀螺仪
        self.acc_proj = nn.Linear(3, d_model)  # 3通道加速度输入
        self.gyro_proj = nn.Linear(3, d_model)  # 3通道陀螺仪输入

        # 加速度LSTM/Transformer层
        acc_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.acc_encoder = nn.TransformerEncoder(acc_encoder_layer, num_layers=num_layers)

        # 陀螺仪LSTM/Transformer层
        gyro_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.gyro_encoder = nn.TransformerEncoder(gyro_encoder_layer, num_layers=num_layers)

        # FFT向量拼接
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),  # 拼接加速度和陀螺仪后的特征
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        输入 x: [B, C=6, T] -> 6通道数据 (加速度+陀螺仪)
        C=6拆分成前3个通道为加速度，后3个通道为陀螺仪
        """
        B, C, T = x.shape

        # 拆分加速度与陀螺仪数据
        acc_input = x[:, :3, :].permute(0, 2, 1)  # [B, T, 3] 传递给加速度编码器
        gyro_input = x[:, 3:, :].permute(0, 2, 1)  # [B, T, 3] 传递给陀螺仪编码器

        # 通过加速度和陀螺仪的Transformer编码器
        acc_out = self.acc_encoder(self.acc_proj(acc_input))  # [B, T, d_model]
        gyro_out = self.gyro_encoder(self.gyro_proj(gyro_input))  # [B, T, d_model]

        # 取每个时间步的最后输出（聚合信息）
        acc_out = acc_out[:, -1, :]  # [B, d_model]
        gyro_out = gyro_out[:, -1, :]  # [B, d_model]

        # 拼接加速度和陀螺仪的输出
        combined = torch.cat([acc_out, gyro_out], dim=-1)  # [B, d_model*2]

        return self.classifier(combined)
