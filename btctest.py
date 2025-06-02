import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load XRP data from CSV
xrp_data = pd.read_csv('Crypto_Data_Csv/XRPUSDT_H1.csv')

# Convert 'datetime' column to datetime object
xrp_data['datetime'] = pd.to_datetime(xrp_data['datetime'])

# Select relevant numeric columns for anomaly detection
columns_to_use = ['open', 'high', 'low', 'close', 'volume']
data = xrp_data[columns_to_use]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA to reduce dimensionality
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(data_scaled)

# Reconstruct the data and calculate reconstruction error
data_reconstructed = pca.inverse_transform(pca.transform(data_scaled))
reconstruction_error = np.mean((data_scaled - data_reconstructed) ** 2, axis=1)

# Determine anomaly threshold using the 95th percentile
threshold = np.percentile(reconstruction_error, 95)

# Detect anomalies
anomalies = reconstruction_error > threshold
xrp_data['Anomaly'] = anomalies

# Print anomaly results
print("Anomaly Detection Results:")
print(xrp_data[['datetime', 'close', 'Anomaly']])

# ==============================
# Optional: Visualization Part
# ==============================

# 1. Plot reconstruction error with threshold line
plt.figure(figsize=(12, 6))
plt.plot(xrp_data['datetime'], reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='95th Percentile Threshold')
plt.title('PCA Reconstruction Error Over Time')
plt.xlabel('Datetime')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Plot detected anomalies on top of close price
anomalies_df = xrp_data[xrp_data['Anomaly']]

plt.figure(figsize=(12, 6))
plt.plot(xrp_data['datetime'], xrp_data['close'], label='XRP Close Price')
plt.scatter(anomalies_df['datetime'], anomalies_df['close'], color='red', label='Anomalies', marker='x')
plt.title('Anomalies Detected in XRP Close Price (PCA-Based)')
plt.xlabel('Datetime')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# Optional: Save results
# ==============================
# xrp_data[['datetime', 'close', 'Anomaly']].to_csv('xrp_anomalies_95th_H1.csv', index=False)
