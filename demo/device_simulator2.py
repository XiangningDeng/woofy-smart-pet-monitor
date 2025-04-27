import time
import json
import random
import pandas as pd
import paho.mqtt.client as mqtt
import numpy as np

#broker = "localhost"
broker = "3.133.137.212"
port = 1883
topic = "pet/activity"
csv_path = "/Users/xiangningdeng/PycharmProjects/DogRing_training/DogMoveData.csv"

# 加载数据
df = pd.read_csv(csv_path)

# 正确脖子特征列
feature_cols = [
    'ANeck_x', 'ANeck_y', 'ANeck_z',
    'GNeck_x', 'GNeck_y', 'GNeck_z'
]
imu_data = df[feature_cols].values
behavior_labels = df["Behavior_1"].values  # ✅ 加载真实行为标签

# 初始化MQTT
client = mqtt.Client()
client.connect(broker, port, 60)
client.loop_start()

while True:
    if not client.is_connected():
        print("❗ MQTT client disconnected. Trying to reconnect...")
        try:
            client.reconnect()
            print("✅ Reconnected successfully!")
        except Exception as e:
            print(f"❌ Reconnect failed: {e}")
            time.sleep(5)
            continue

    # 随机选一个合法的起点
    max_start = len(imu_data) - 1200

    valid_batch_found = False
    attempt = 0

    while not valid_batch_found:
        start_idx = random.randint(0, max_start)
        selected_labels = behavior_labels[start_idx : start_idx + 1200]

        if all(lab.lower() != "<undefined>" for lab in selected_labels):
            valid_batch_found = True
        else:
            attempt += 1
            if attempt >= 20:
                print("⚠️ Warning: Couldn't find clean 1200 frames after 20 tries. Sending anyway (may include <undefined>)")
                valid_batch_found = True

    batch = []
    current_time = time.time()

    selected_imu = imu_data[start_idx : start_idx + 1200]
    selected_labels = behavior_labels[start_idx : start_idx + 1200]

    # 统计这一段主要是什么动作
    labels_unique, counts = np.unique(selected_labels, return_counts=True)
    dominant_label = labels_unique[np.argmax(counts)]

    for i, row in enumerate(selected_imu):
        t = current_time + i * 0.01

        payload = {
            "device_id": "Dog002",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
            "neck_acc_x": float(row[0]),
            "neck_acc_y": float(row[1]),
            "neck_acc_z": float(row[2]),
            "neck_gyro_x": float(row[3]),
            "neck_gyro_y": float(row[4]),
            "neck_gyro_z": float(row[5])
        }
        batch.append(payload)

    payload_str = json.dumps(batch)

    result = client.publish(topic, payload_str, qos=1)
    status = result[0]

    if status == 0:
        print(f"✅ Sent a batch of {len(batch)} frames (start_idx={start_idx}) | 🏷️ True behavior: {dominant_label}")
    else:
        print(f"❌ Failed to send message with status {status}")

    time.sleep(12)
