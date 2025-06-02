import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features





class MyDataset(Dataset):
    def __init__(self, args, root_path=None, flag='train', data_path='MyDataset.csv', target='label', scale=True, win_size=12, step=1):
        # Extract the parameters from 'args'
        flag = flag.lower()
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        # Add win_size for sliding window size
        self.win_size = win_size
        self.step = step
        # Ensure flag is one of 'train', 'test', or 'val'
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # If root_path is provided in args, use it, else set to None
        if root_path is not None:
            self.root_path = root_path
        elif hasattr(args, 'root_path'):
            self.root_path = args.root_path  
        else:
            self.root_path = None

        self.data_path = data_path
        self.target = target
        self.scale = scale

        # Load and preprocess data
        self.__read_data__()

        # Set max_seq_len to the sequence length, or calculate it dynamically if needed
        self.max_seq_len = self.seq_len  # or use other logic to set the max sequence length

        # Extract unique class names from the 'label' column
        self.class_names = sorted(self.df_raw[self.target].unique())  # Use df_raw from self

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Read data
        full_path = os.path.join(self.root_path, self.data_path)
        self.df_raw = pd.read_csv(full_path)

        # Check if 'date' column exists
        has_date = 'datetime' in self.df_raw.columns

        # Define train/val/test splits
        border1s = [0, len(self.df_raw) // 3, 2 * len(self.df_raw) // 3]
        border2s = [len(self.df_raw) // 3, 2 * len(self.df_raw) // 3, len(self.df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Extract features (exclude label and optionally date)
        if has_date:
            feature_cols = self.df_raw.columns.difference(['label', 'date'])
        else:
            feature_cols = self.df_raw.columns.difference(['label'])

        df_data = self.df_raw[feature_cols]
        self.feature_df = df_data

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = self.df_raw[self.target].values[border1:border2]

        # Prepare time features if date is available
        if has_date:
            df_stamp = pd.to_datetime(self.df_raw['datetime'])
            data_stamp = time_features(df_stamp, timeenc=0)
            self.data_stamp = data_stamp[border1:border2]
        else:
            self.data_stamp = None

    def __len__(self):
        
        if self.set_type == "train":
            return (self.data_x.shape[0] - self.win_size) // self.step + 1
        elif self.set_type == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.set_type == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data_x.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step

        x = np.float32(self.data_x[index:index + self.win_size])
        y = np.float32([self.data_y[index + self.win_size - 1]])
        y = np.reshape(y, (1,))
        padding_mask = np.ones(self.win_size, dtype=np.float32)

        if self.data_stamp is not None:
            # Use provided timestamp embeddings
            x_mark_enc = np.float32(self.data_stamp[index:index + self.win_size])
        else:
            # Hardcoded normalized time embedding: [0.0, ..., 1.0]
            x_mark_enc = np.linspace(0, 1, self.win_size).reshape(self.win_size, 1).astype(np.float32)

        return x, y, padding_mask, x_mark_enc





    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
