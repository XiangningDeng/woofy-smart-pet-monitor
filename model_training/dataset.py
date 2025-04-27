import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import config  # 引入 config.py 的 USE_FFT 参数

def compute_fft_features(x_window):
    """Compute mean FFT features for each IMU channel."""
    fft_feats = np.abs(np.fft.rfft(x_window, axis=0))  # (freq_bins, channels)
    fft_mean = np.mean(fft_feats, axis=0)              # (channels,)
    return fft_mean

# ✅ 最新版类别合并映射表（Panting也归到Stilling）
CLASS_MERGE_MAP = {
    "Eating": "Feeding",
    "Drinking": "Feeding",
    "Sitting": "Stilling",
    "Lying chest": "Stilling",
    "Pacing": "Stilling",
    "Standing": "Stilling",
    "Panting": "Stilling",               # ✅ Panting也归Stilling
    "Galloping": "Running",
    "Jumping": "Running",
    "Trotting": "Running",
    "Carrying object": "Playing",
    "Bowing": "Playing",
    "Tugging": "Playing",
    "Playing": "Playing",
    "Shaking": "Playing",
    "Sniffing": "Sniffing",
    "Extra_Synchronization": "Running",
    "Synchronization": "Walking",
    "Walking": "Walking"
}

class DogBehaviorDataset(Dataset):
    def __init__(self, csv_path, window_size=200, step=100):
        self.window_size = window_size
        self.step = step

        # === 加载数据 ===
        df = pd.read_csv(csv_path)
        df = df[df['Behavior_1'].str.lower() != '<undefined>'].copy()
        df = df.reset_index(drop=True)

        # === 标签合并两次，确保归并干净 ===
        df["Behavior_1_Merged"] = df["Behavior_1"].map(CLASS_MERGE_MAP).fillna(df["Behavior_1"])
        df["Behavior_1_Merged"] = df["Behavior_1_Merged"].map(CLASS_MERGE_MAP).fillna(df["Behavior_1_Merged"])

        # === 只取 Neck 部位的IMU特征列 ===
        feature_cols = [col for col in df.columns if 'Neck' in col]
        self.features = df[feature_cols].values.astype(np.float32)  # (total_frames, 6)

        # === 标签编码 ===
        le = LabelEncoder()
        self.labels = le.fit_transform(df["Behavior_1_Merged"].values)
        self.label_encoder = le

        # === 打印标签归并情况 ===
        print("🔍 Behavior Label Merge Summary:")
        for raw in sorted(df["Behavior_1"].unique()):
            mapped = CLASS_MERGE_MAP.get(raw, raw)
            print(f"  {raw:<25} ➜  {mapped}")
        print(f"\n✅ Final Classes: {list(le.classes_)}\n")

        # === 滑动窗口切片 ===
        self.X, self.Y = [], []
        for i in range(0, len(self.features) - window_size, step):
            x_win = self.features[i:i+window_size]
            y_win = self.labels[i:i+window_size]
            y_mode = np.bincount(y_win).argmax()  # 多数表决

            if config.USE_FFT:
                fft_feat = compute_fft_features(x_win)
                x_aug = np.vstack([x_win, fft_feat[np.newaxis, :]])  # (window_size+1, channels)
            else:
                x_aug = x_win

            self.X.append(x_aug)
            self.Y.append(y_mode)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



#CLASS_MERGE_MAP = {
    #"Eating": "Feeding",
    #"Drinking": "Feeding",
    #"Sitting": "Stilling",              # ✅ Sitting归到Stilling
    #"Lying chest": "Stilling",           # ✅ Lying chest归到Stilling
    #"Pacing": "Stilling",                # ✅ Pacing归到Stilling
    #"Standing": "Stilling",              # ✅ Standing归到Stilling
    #"Galloping": "Running",
    #"Jumping": "Running",
    #"Trotting": "Running",
    #"Carrying object": "Playing",         # ✅ 原Other → Playing
    #"Bowing": "Playing",                  # ✅ 原Other → Playing
    #"Tugging": "Playing",
    #"Playing": "Playing",
    #"Shaking": "Playing",                 # ✅ Shaking也归Playing
    #"Panting": "Panting",
    #"Sniffing": "Sniffing",
    #"Extra_Synchronization": "Running",   # ✅ Extra_Sync归到Running
    #"Synchronization": "Walking",         # ✅ Sync归到Walking
    #"Walking": "Walking"
#}



