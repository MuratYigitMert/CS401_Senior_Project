import matplotlib.pyplot as plt
import numpy as np

# Replace these with your actual metric values from your evaluation printouts
model_names = ['LSTMODetect', 'Isolation Forest', 'LODA', 'LOF']
precision = [0.0041, 0.0270, 0.0611, 0.0054]
recall =    [0.0229, 0.1499, 0.3386, 0.0265]
f1 =        [0.0070, 0.0458, 0.1035, 0.0089]
accuracy =  [0.8828, 0.8874, 0.8943, 0.8941]
f_beta =    [0.4724, 0.4930, 0.5237, 0.4765]  # If beta=1, same as F1

# Set up x-axis positions and bar width
x = np.arange(len(model_names))
width = 0.15

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 2*width, precision, width, label='Precision')
ax.bar(x - width, recall, width, label='Recall')
ax.bar(x, f1, width, label='F1 Score')
ax.bar(x + width, accuracy, width, label='Accuracy')
ax.bar(x + 2*width, f_beta, width, label='F1 Macro')

# Set labels and title
ax.set_ylabel('Score')
ax.set_title('Performance of Anomaly Detection Models (Lee–Mykland Ground Truth)')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend(loc='upper right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
