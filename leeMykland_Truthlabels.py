import pandas as pd
import numpy as np


df = pd.read_csv('Crypto_Data_Csv/SOLUSDT_H1.csv')


def lee_mykland_jumps(df):
   
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))

 
    mean_log_return = df['log_return'].mean()
    std_log_return = df['log_return'].std()


    df['jump'] = (df['log_return'].abs() > (mean_log_return + 3 * std_log_return)).astype(int)

    return df


df_lstm_labeled = lee_mykland_jumps(df)


df_lstm_labeled.to_csv('Anomaly_lee_results/SOLUSDT_H1_LeeMykland_TruthLabels.csv', index=False)
