import pandas as pd
import matplotlib.pyplot as plt

# Load the original ETH/USDT data (make sure datetime column is present)
df = pd.read_csv('Crypto_Data_Csv/ETHUSDT_H1.csv')
df['open_time'] = pd.to_datetime(df['datetime'])  # Adjust if your datetime column is named differently

# Load saved LODA results
result_df = pd.read_csv('FinalReportFiles/ETHUSDT_H1_LODA_results.csv')

# Align indices
df = df.loc[result_df.index].reset_index(drop=True)
result_df = result_df.reset_index(drop=True)

# Add anomaly labels
df['Anomaly'] = result_df['TODS.anomaly_detection_primitives.LODAPrimitive0_0']

# Plot Close price with anomalies
plt.figure(figsize=(14,6))
plt.plot(df['open_time'], df['close'], label='ETH Close Price')
plt.scatter(df[df['Anomaly'] == 1]['open_time'],
            df[df['Anomaly'] == 1]['close'],
            color='red', label='Detected Anomalies', marker='x')
plt.title('ETH Price with Anomalies Detected by LODA (Loaded Results)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
