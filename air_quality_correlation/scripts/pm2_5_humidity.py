import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host='192.168.50.200',
        user='juanD',
        password='JuanD1_SMART@NYUAD',
        database='home_assistant'
    )

# Fetch data from the database
def fetch_data():
    conn = connect_to_db()
    cursor = conn.cursor()

    # Fetch PM 2.5 data during working hours (8 AM to 5 PM)
    query_pm25 = """
    SELECT AVG(sensor_statistics_hourly.average) AS avg_pm25, 
           FROM_UNIXTIME(sensor_statistics_coorelation.timestamp, '%Y-%m-%d %H:00:00') AS hour
    FROM sensor_statistics_hourly
    JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
    WHERE sensor_statistics_hourly.metadata_id = 86 AND HOUR(FROM_UNIXTIME(sensor_statistics_coorelation.timestamp)) >= 8 AND HOUR(FROM_UNIXTIME(sensor_statistics_coorelation.timestamp)) < 17
    GROUP BY hour
    ORDER BY hour
    """
    cursor.execute(query_pm25)
    pm25_results = cursor.fetchall()

    # Fetch humidity data during working hours (8 AM to 5 PM)
    query_humidity = """
    SELECT AVG(sensor_statistics_hourly.average) AS avg_humidity, 
           FROM_UNIXTIME(sensor_statistics_coorelation.timestamp, '%Y-%m-%d %H:00:00') AS hour
    FROM sensor_statistics_hourly
    JOIN sensor_statistics_coorelation ON sensor_statistics_hourly.update_time_id = sensor_statistics_coorelation.update_time_id
    WHERE sensor_statistics_hourly.metadata_id = 18 AND HOUR(FROM_UNIXTIME(sensor_statistics_coorelation.timestamp)) >= 8 AND HOUR(FROM_UNIXTIME(sensor_statistics_coorelation.timestamp)) < 17
    GROUP BY hour
    ORDER BY hour
    """
    cursor.execute(query_humidity)
    humidity_results = cursor.fetchall()

    cursor.close()
    conn.close()

    return pm25_results, humidity_results

# Retrieve and merge data
pm25_results, humidity_results = fetch_data()

# Convert to DataFrame
df_pm25 = pd.DataFrame(pm25_results, columns=['avg_pm25', 'hour'])
df_humidity = pd.DataFrame(humidity_results, columns=['avg_humidity', 'hour'])

# Convert hour to datetime
df_pm25['hour'] = pd.to_datetime(df_pm25['hour'])
df_humidity['hour'] = pd.to_datetime(df_humidity['hour'])

# Merge dataframes on the hour column
df_merged = pd.merge(df_pm25, df_humidity, on='hour', how='inner')

# Calculate Pearson correlation
correlation, _ = pearsonr(df_merged['avg_pm25'], df_merged['avg_humidity'])
print(f"Pearson correlation coefficient between PM 2.5 and Humidity: {correlation}")

# Plotting with line of best fit
plt.figure(figsize=(10, 6))
sns.regplot(x='avg_pm25', y='avg_humidity', data=df_merged, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
plt.title('Correlation between PM 2.5 and Humidity with Line of Best Fit During Working Hours')
plt.xlabel('Average PM 2.5')
plt.ylabel('Average Humidity')
plt.grid(True)
plt.show()
