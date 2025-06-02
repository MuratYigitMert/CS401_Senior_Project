import pandas as pd
import matplotlib.pyplot as plt

# Load the original data (make sure 'Open time' column exists)
df = pd.read_csv('Crypto_Data_Csv/btc_1h_data_2018_to_2025.csv')
df['Open time'] = pd.to_datetime(df['Open time'])

# Load the saved anomaly results from LSTMODetect
result_df = pd.read_csv('Anomaly_Results_WithTods/BTC_1H_LSTM.csv')

# Align indices if needed (assuming row order matches)
df = df.loc[result_df.index].reset_index(drop=True)
result_df = result_df.reset_index(drop=True)

# Add anomaly labels to dataframe
df['Anomaly'] = result_df['LSTMODetector0_0']  # Adjust column name if different

# Plot Close price with anomalies
plt.figure(figsize=(14,6))
plt.plot(df['Open time'], df['Close'], label='BTC Close Price')
plt.scatter(df[df['Anomaly'] == 1]['Open time'],
            df[df['Anomaly'] == 1]['Close'],
            color='red', label='Detected Anomalies', marker='x')
plt.title('BTC Price with Anomalies Detected by LSTMODetect (Loaded Results)')
plt.xlabel('Open time')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
