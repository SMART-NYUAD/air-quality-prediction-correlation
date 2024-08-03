import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

    # Fetch raw people count data
    query_people = """
    SELECT time, people_count 
    FROM people_number
    WHERE HOUR(FROM_UNIXTIME(time)) >= 8 AND HOUR(FROM_UNIXTIME(time)) < 17
    ORDER BY time
    """
    cursor.execute(query_people)
    people_results = cursor.fetchall()

    cursor.close()
    conn.close()

    return pm25_results, people_results

# Process people count data to calculate hourly people time density
def process_people_count_data(people_results):
    people_count_df = pd.DataFrame()
    times = [datetime.fromtimestamp(entry[0]) for entry in people_results]
    people_count_df['timestamp'] = pd.Series(times)
    people_count_df['people_number'] = pd.to_numeric([entry[1] for entry in people_results], errors='coerce')

    # Set the index to `timestamp`
    people_count_df.set_index('timestamp', inplace=True)

    # Insert rows for the first and last second of each hour
    start_of_hours = pd.date_range(start=people_count_df.index.min().floor('h'),
                                   end=people_count_df.index.max().floor('h'),
                                   freq='h')
    start_of_hours_df = pd.DataFrame(index=start_of_hours)  # Create DataFrame with index as timestamps
    end_of_hours = start_of_hours + timedelta(hours=1) - timedelta(milliseconds=1)
    end_of_hours_df = pd.DataFrame(index=end_of_hours)  # Create DataFrame with index as timestamps

    # Concatenate DataFrames
    people_count_copied = pd.concat([people_count_df, start_of_hours_df, end_of_hours_df]).sort_index().copy()

    # Forward fill the `people_number` column
    people_count_copied['people_number'] = people_count_copied['people_number'].ffill()

    # Calculate time differences and people time density
    people_count_copied['time_diff'] = people_count_copied.index.to_series().shift(-1).diff().dt.total_seconds().fillna(0) / 3600
    people_count_copied['people_time_density'] = people_count_copied['people_number'] * people_count_copied['time_diff']

    # Resample by hour and calculate the sum of `people_time_density` for each hour
    hourly_people_time_density_sum = people_count_copied['people_time_density'].resample('H').sum()

    return hourly_people_time_density_sum

# Retrieve and merge data
pm25_results, people_results = fetch_data()

# Process the people count data
hourly_people_time_density_sum = process_people_count_data(people_results)

# Convert to DataFrame
df_pm25 = pd.DataFrame(pm25_results, columns=['avg_pm25', 'hour'])
df_pm25['hour'] = pd.to_datetime(df_pm25['hour'])
df_people = pd.DataFrame(hourly_people_time_density_sum).reset_index()
df_people.columns = ['hour', 'people_time_density']

# Merge dataframes on the hour column
df_merged = pd.merge(df_pm25, df_people, on='hour', how='inner')

# Calculate Pearson correlation
correlation, _ = pearsonr(df_merged['avg_pm25'], df_merged['people_time_density'])
print(f"Pearson correlation coefficient between PM 2.5 and People Time Density: {correlation}")

# Plotting with line of best fit
plt.figure(figsize=(10, 6))
sns.regplot(x='avg_pm25', y='people_time_density', data=df_merged, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
plt.title('Correlation between PM 2.5 and People Time Density with Line of Best Fit During Working Hours')
plt.xlabel('Average PM 2.5')
plt.ylabel('People Time Density')
plt.grid(True)
plt.show()
