import torch
import torch.nn as nn

class DogBehaviorBiLSTM_Fusion(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(DogBehaviorBiLSTM_Fusion, self).__init__()

        # 3加速度通道
        self.acc_lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 3陀螺仪通道
        self.gyro_lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 全局融合后的大BiLSTM
        self.fusion_lstm = nn.LSTM(
            input_size=hidden_dim * 4,  # acc(2xhidden) + gyro(2xhidden)
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # BiLSTM是双向的
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        输入: (B, C=6, T)
        拆分: (B, 3, T) + (B, 3, T)
        """
        B, C, T = x.shape
        acc = x[:, :3, :].permute(0, 2, 1)    # (B, T, 3)
        gyro = x[:, 3:, :].permute(0, 2, 1)   # (B, T, 3)

        # 分别通过小BiLSTM
        acc_out, _ = self.acc_lstm(acc)       # (B, T, hidden_dim*2)
        gyro_out, _ = self.gyro_lstm(gyro)     # (B, T, hidden_dim*2)

        # 拼接加速度+陀螺仪特征
        combined = torch.cat([acc_out, gyro_out], dim=-1)  # (B, T, hidden_dim*4)

        # 送入大BiLSTM
        fusion_out, _ = self.fusion_lstm(combined)         # (B, T, hidden_dim*2)

        # 取最后一个时间步
        out = fusion_out[:, -1, :]  # (B, hidden_dim*2)

        out = self.dropout(out)
        out = self.classifier(out)

        return out
