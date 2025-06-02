import pandas as pd

df = pd.read_csv("/home/murat/tods/Anomaly_lee_results/SOLUSDT_H1_LeeMykland_TruthLabels.csv")

# Drop datetime

df = df.drop(columns=["log_return"])

# Rename jump to label
df = df.rename(columns={"jump": "label"})

# Optionally drop log_return if not needed
# df = df.drop(columns=["log_return"])

# Save to the right location
df.to_csv("/home/murat/tods/Time-Series-Library/dataset/MyDataset/MyDatasetWITHdate.csv", index=False)
print("done")
