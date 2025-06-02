import pandas as pd
import matplotlib.pyplot as plt

# Load the original ETH price data
eth_df = pd.read_csv('Crypto_Data_Csv/ETHUSDT_H1.csv')
eth_df['datetime'] = pd.to_datetime(eth_df['datetime'])

# Clean and align with detection input (ensure same order as used for TODS)
eth_df = eth_df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna().reset_index(drop=True)

# Load the Isolation Forest detection results
result_df = pd.read_csv('FinalReportFiles/ETHUSDT_H1_IsolationForest_results.csv').reset_index(drop=True)

# Attach anomaly results to original DataFrame
eth_df['Anomaly'] = result_df['TODS.anomaly_detection_primitives.IsolationForest0_0']  # Adjust column name if needed

# Plot close price with anomalies
plt.figure(figsize=(14, 6))
plt.plot(eth_df['datetime'], eth_df['close'], label='ETH Close Price')
plt.scatter(eth_df[eth_df['Anomaly'] == 1]['datetime'],
            eth_df[eth_df['Anomaly'] == 1]['close'],
            color='red', label='Detected Anomalies', marker='x')
plt.title('ETH Price with Anomalies Detected by Isolation Forest (TODS)')
plt.xlabel('Datetime')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
