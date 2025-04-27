import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# === Page settings
st.set_page_config(page_title="Pet Activity Dashboard", page_icon="🐾", layout="centered")
st.title("🐾 Pet Activity Monitoring Dashboard")

# === Auto-refresh every 12 seconds
st_autorefresh(interval=12 * 1000, key="data_refresh")

# === Connect to database
conn = sqlite3.connect("/Users/xiangningdeng/PycharmProjects/DogRing_training/pet_activity.db")
query = "SELECT * FROM activity_log"
df = pd.read_sql_query(query, conn)

if df.empty:
    st.warning("No activity data available...")
else:
    # === Select device_id
    device_ids = df["device_id"].unique()
    selected_device = st.selectbox("Select Device ID:", options=device_ids)

    df_device = df[df["device_id"] == selected_device]

    if df_device.empty:
        st.warning(f"No activity data for {selected_device}")
    else:
        st.subheader(f"📊 Total Activity Duration for {selected_device}")

        # Each record represents 4 seconds
        seconds_per_record = 4

        # Calculate total duration
        activity_counts = df_device["activity"].value_counts()
        activity_seconds = activity_counts * seconds_per_record

        # Decide unit based on max duration
        max_seconds = activity_seconds.max()

        if max_seconds < 60:
            unit = "seconds"
            display_values = activity_seconds
        elif max_seconds < 3600:
            unit = "minutes"
            display_values = activity_seconds / 60
        else:
            unit = "hours"
            display_values = activity_seconds / 3600

        # === Unified bar chart
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(display_values.index, display_values.values, color=plt.cm.tab20.colors[:len(display_values)])

        ax.set_ylabel(f"Duration ({unit})")
        ax.set_xlabel("Activity Type")
        ax.set_title(f"Total Duration of Each Activity - {selected_device}")
        plt.xticks(rotation=45)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        st.pyplot(fig)

        # === Pie chart
        st.subheader(f"🥧 Activity Proportion for {selected_device}")

        fig2, ax2 = plt.subplots(figsize=(8, 6))

        wedges, texts = ax2.pie(
            activity_counts,
            labels=None,
            startangle=90
        )

        ax2.legend(
            wedges,
            labels=[f"{act} ({pct:.1f}%)" for act, pct in
                    zip(activity_counts.index, 100 * activity_counts.values / activity_counts.values.sum())],
            title="Activities",
            loc="lower right",
            bbox_to_anchor=(1.3, 0.05)
        )

        ax2.axis('equal')
        st.pyplot(fig2)

        # === Latest activity records
        st.subheader(f"🕒 Recent Activity Logs for {selected_device}")
        st.dataframe(df_device.tail(20))
