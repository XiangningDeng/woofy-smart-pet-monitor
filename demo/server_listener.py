import paho.mqtt.client as mqtt
import json
import sqlite3
from datetime import datetime
import time

broker = "localhost"
port = 1883
topic = "pet/device1/data"
db_file = "pet_activity.db"

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"✅ Received payload: {payload}")

        weight = 10  # 假设10kg
        duration_hr = 30 / 3600  # 30秒运动

        base_met = 1.0 + payload["acc_energy"]
        kcal = base_met * weight * duration_hr

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dog_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            timestamp TEXT,
            acc_energy REAL,
            kcal REAL
        )
        """)
        conn.commit()

        cursor.execute("""
        INSERT INTO dog_activity (device_id, timestamp, acc_energy, kcal)
        VALUES (?, ?, ?, ?)
        """, (
            payload["device_id"],
            payload["timestamp"],
            payload["acc_energy"],
            kcal
        ))
        conn.commit()
        conn.close()

        print(f"💾 Saved to DB: kcal={kcal:.3f}")

    except Exception as e:
        print(f"❌ Error processing message: {e}")

def on_disconnect(client, userdata, rc):
    print(f"⚠️ Disconnected from MQTT broker with code {rc}")
    while True:
        try:
            print("🔄 Trying to reconnect...")
            client.reconnect()
            print("✅ Reconnected to MQTT broker!")
            client.subscribe(topic, qos=1)
            break
        except:
            print("❗ Reconnect failed, retrying in 5 seconds...")
            time.sleep(5)

def main():
    client = mqtt.Client()
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect(broker, port, keepalive=60)
    client.subscribe(topic, qos=1)
    print("✅ Subscribed to MQTT topic. Waiting for data...")
    client.loop_forever()

if __name__ == "__main__":
    main()
