import torch.nn as nn

class DogBehaviorLSTM(nn.Module):
    def __init__(self, input_channels=12, num_classes=5, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C)，注意LSTM要time在第二维
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 只取最后一个时间步
        out = self.fc(out)
        return out
