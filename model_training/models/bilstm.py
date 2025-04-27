import torch
import torch.nn as nn

class DogBehaviorBiLSTM(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(DogBehaviorBiLSTM, self).__init__()

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=input_channels,  # 输入通道数（Neck IMU数量）
            hidden_size=hidden_dim,     # 每层隐藏维度
            num_layers=num_layers,      # 堆叠多少层
            batch_first=True,           # (B, T, C)
            bidirectional=True          # 双向
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),   # *2因为是双向
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        输入: (B, C, T) => 转成 (B, T, C)
        """
        x = x.permute(0, 2, 1)  # (B, T, C)

        out, _ = self.lstm(x)  # out: (B, T, hidden_dim*2)

        # 取最后一个时间步（或取平均池化，按需求改）
        out = out[:, -1, :]  # (B, hidden_dim*2)

        out = self.dropout(out)
        out = self.classifier(out)

        return out
