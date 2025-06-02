from tods.tods.detection_algorithm.PyodIsolationForest import IsolationForestPrimitive
from d3m import container
import pandas as pd

csv_path = 'crypto_data/ETHUSDT_H1.csv'

df = pd.read_csv(csv_path)

df = df[['open', 'high', 'low', 'close', 'volume']].dropna() 

df_d3m = container.DataFrame(df)

hyperparams = {
    'window_size': 50,
    'step_size' : 1,
    'metric_params': None,
    'use_semantic_types': False,
    'return_subseq_inds': False,
    'return_semantic_type': None,
    'return_result': True,
    'add_index_columns': False,
    'return_result': 'replace',
    'contamination': 0.1,
    'n_estimators': 100,
    'max_samples': 'auto',
    'max_features': 1,
    'bootstrap': False,
    'behaviour': 'old',
    'random_state': None,
    'verbose': 0
}

IsoForest = IsolationForestPrimitive(hyperparams=hyperparams)

IsoForest.set_training_data(inputs=df_d3m)

IsoForest.fit()

result = IsoForest.produce(inputs=df_d3m)

result_df = result.value

result_csv_path = 'results/ETHUSDT_H1_IsolationForest_results.csv'

result_df.to_csv(result_csv_path, index=False)