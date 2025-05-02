from flask import Flask, jsonify
import sqlite3
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ 允许跨域请求

DB_PATH = "pet_activity.db"

# === 数据库连接工具 ===
def get_connection():
    return sqlite3.connect(DB_PATH)

# === 路由定义 ===

# 获取最新一条行为数据
@app.route("/api/latest", methods=["GET"])
def get_latest_activity():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row:
        return jsonify({
            "device_id": row[1],
            "timestamp": row[2],
            "activity": row[3]
        })
    else:
        return jsonify({"message": "No data"}), 404

# 获取所有行为数据
@app.route("/api/all", methods=["GET"])
def get_all_activities():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM activity_log ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    data = [
        {
            "device_id": r[1],
            "timestamp": r[2],
            "activity": r[3]
        }
        for r in rows
    ]
    return jsonify(data)

# 健康检查
@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

# === 启动服务 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)