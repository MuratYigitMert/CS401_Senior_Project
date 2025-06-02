import pandas as pd
import matplotlib.pyplot as plt

# Load original ETH data (ensure datetime column is correct)
df = pd.read_csv('Crypto_Data_Csv/ETHUSDT_H1.csv')
df['open_time'] = pd.to_datetime(df['datetime'])  # Adjust if named differently

# Load saved LOF results
result_df = pd.read_csv('Roshaan_Anomaly_Results/SOLUSDT_H1_LOF_results.csv')

# Align indices
df = df.loc[result_df.index].reset_index(drop=True)
result_df = result_df.reset_index(drop=True)

# Add anomaly labels
df['Anomaly'] = result_df['TODS.anomaly_detection_primitives.LOFPrimitive0_0']

# Plot Close price with anomalies
plt.figure(figsize=(14,6))
plt.plot(df['open_time'], df['close'], label='ETH Close Price')
plt.scatter(df[df['Anomaly'] == 1]['open_time'],
            df[df['Anomaly'] == 1]['close'],
            color='red', label='Detected Anomalies', marker='x')
plt.title('ETH Price with Anomalies Detected by LOF (Loaded Results)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
