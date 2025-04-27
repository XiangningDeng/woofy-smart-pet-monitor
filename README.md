# woofy-smart-pet-monitor

## 🐾 Pet Behavior Recognition - Model Development and Edge Deployment

This repository documents the complete pipeline from **pet behavior classification model development** to **local server deployment** for real-time activity monitoring.

---

## Project Overview

This project focuses on:
- Predicting pet activities from IMU data (6-axis: 3-axis accelerometer + 3-axis gyroscope)
- Deploying a real-time, lightweight inference server
- Local database logging and visualization through a web dashboard

---

## 1. Behavior Recognition Model

### Data Preparation
- **Source**: DogMoveData.csv (public dataset, available at https://pmc.ncbi.nlm.nih.gov/articles/PMC8777071/)
- **Input Features**: Neck IMU signals (ANeck_x, ANeck_y, ANeck_z, GNeck_x, GNeck_y, GNeck_z)
- **Sampling Rate**: 100Hz
- **Window Size**: 400 frames (approx. 4 seconds)
- **Step Size**: 100 frames

### Label Engineering
- Merged similar behaviors to improve model performance:
  - Standing + Resting → Stilling
  - Tugging, Bowing, Carrying object, Shaking → Playing
  - Synchronization → Walking
- Filtered out `<undefined>` behavior segments during training and simulation.

### Feature Engineering
- **FFT Feature Concatenation**: For each 400-frame window, we computed the mean magnitude of the FFT spectrum for each channel. These frequency-domain features (6 additional values) were concatenated with the raw time-domain features to enrich the signal representation.
- **Sensor-Specific Encoding**: The accelerometer (3 channels) and gyroscope (3 channels) data were first separately encoded using parallel branches (small LSTM/Transformer blocks), then fused at a later stage to emphasize modality-specific dynamics before combining into a unified behavior embedding.

### Model Architecture
- **Input Shape**: (batch_size, 6 channels, 400 time steps)
- **Structure**:
  1. Separate Encoder Blocks for Accelerometer and Gyroscope (shared architecture)
  2. Concatenation of encoded accelerometer and gyroscope outputs
  3. Transformer Encoder for fused temporal modeling
  4. Global average pooling across time dimension
  5. Concatenation with FFT features
  6. MLP classifier head with ReLU + Dropout layers
- **Framework**: PyTorch 2.x (MPS acceleration enabled on Mac M1/M2)

#### Model Diagram (Simplified)

```text
          +---------------------+        +---------------------+
          |  Accelerometer (3D)  |        |   Gyroscope (3D)     |
          +----------+----------+        +----------+----------+
                     |                            |
           Small Encoder (LSTM/Transformer) Small Encoder
                     |                            |
           +---------+----------------------------+---------+
                     |       Concatenation (Fusion)           |
                     +----------------------------------------+
                                       |
                            Transformer Encoder
                                       |
                            Global Average Pooling
                                       |
                        [Concat FFT Mean Spectrum]
                                       |
                               MLP Classifier
                                       |
                                  Output Class
```

### Performance
- Cross-validated accuracy ~82%
- Good generalization across merged behavior classes.

![confusion_matrix.png](confusion_matrix.png)

---

##  2. Local Server Prototype

### System Components

| Module | Description |
|---|---|
| MQTT Broker | Mosquitto broker running locally (port 1883) |
| Simulator | Python script sending IMU batches every 12 seconds (1200 frames) |
| Listener & Inference Server | Python script subscribing to MQTT topic, inferring behavior every 4s window |
| SQLite Database | Lightweight database logging prediction results |
| Streamlit Dashboard | Real-time visualization of predicted behaviors |

### Data Flow

```
[Simulator: 1200 frames/12s] ➔ [MQTT: pet/activity topic] ➔ [Listener: predict every 400 frames] ➔ [SQLite: log activity] ➔ [Dashboard: refresh view]
```

### MQTT Topic Specification
- **Topic**: `pet/activity`
- **Payload Format**:
```json
{
  "device_id": "Dog001",
  "timestamp": "2025-04-26 04:35:08",
  "neck_acc_x": 0.12,
  "neck_acc_y": 0.04,
  "neck_acc_z": 9.81,
  "neck_gyro_x": 0.5,
  "neck_gyro_y": -0.2,
  "neck_gyro_z": 0.1
}
```
- **Batch**: 1200 frames sent together every 12 seconds


---

##  3. Streamlit Dashboard

- Displays total duration of each activity.
- Auto-refreshes every 12 seconds.
- Time unit auto-scaling (seconds → minutes → hours) based on duration.
- Activity proportion visualized via pie chart.
- Latest behavior predictions listed in a dynamic table.

**Access**:
```
http://localhost:8501
```

---

##  4. Deployment Notes

### Requirements
- Python 3.10+
- PyTorch (with MPS for Mac acceleration)
- Streamlit
- paho-mqtt
- Mosquitto Broker

### Running Order

```bash
# 1. Start Mosquitto MQTT Broker
brew services start mosquitto

# 2. Start server listener
python server_listener.py

# 3. Start simulator
python pet_simulator.py

# 4. Launch dashboard
streamlit run pet_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Full Running Procedure

1. **Start Mosquitto MQTT Broker**:
   - Ensure port 1883 is open.
   - Acts as a message broker between simulator and listener.

2. **Start the Server Listener**:
   - Subscribes to `pet/activity` topic.
   - Waits for incoming IMU batches.
   - Splits each 1200-frame batch into three 400-frame windows.
   - Runs model inference on each window.
   - Saves the predicted activity with timestamp into SQLite database.

3. **Start the IMU Data Simulator**:
   - Publishes a batch of 1200 frames every 12 seconds.
   - Payloads mimic real IMU sensor readings.

4. **Launch Streamlit Dashboard**:
   - Periodically reads from SQLite database.
   - Updates activity distribution, proportions, and recent activity records automatically.

5. **Access Dashboard**:
   - Open browser and navigate to `http://localhost:8501`

6. **Monitor Logs**:
   - View simulator sending status, server listener predictions, and dashboard UI.


---

##  5. Future Improvements

- Real device integration (ESP32-based IMU wearable)
- Move MQTT broker and server to cloud (Aliyun / AWS)
- Model quantization / ONNX export for lower edge inference cost
- Edge-side lightweight NTP-synchronized timestamp handling

