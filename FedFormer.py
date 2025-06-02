import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load full dataset
df = pd.read_csv('Time-Series-Library/dataset/MyDataset/MyDataset.csv')

# Create datetime if missing
if 'datetime' not in df.columns:
    df['datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')

# Load predicted probabilities for FEDformer
probs = np.load('./results/FEDformer/FEDformer_probs.npy')  # adjust path if needed
preds = (probs > 0.5).astype(int)

# Trim df to align with preds
start_index = len(df) - len(preds)
df_trimmed = df.iloc[start_index:].reset_index(drop=True)

# Assign predictions
df_trimmed['pred'] = preds

# Create save directory
save_dir = '/home/murat/Pictures/Screenshots'
os.makedirs(save_dir, exist_ok=True)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_trimmed.index, df_trimmed['close'], label='Close Price')
plt.scatter(df_trimmed.index[df_trimmed['pred'] == 1],
            df_trimmed.loc[df_trimmed['pred'] == 1, 'close'],
            color='red', marker='x', label='Predicted Anomalies')
plt.xlabel("Sample Number")
plt.ylabel("Close Price")
plt.title("FEDformer Anomaly Detection on Price Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "fedformer_price_with_anomalies.png"))
plt.close()
