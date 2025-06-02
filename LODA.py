from tods.tods.detection_algorithm.PyodLODA import LODAPrimitive
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
    'n_bins': 10,
    'n_random_cuts': 100,
}

loda = LODAPrimitive(hyperparams=hyperparams)

loda.set_training_data(inputs=df_d3m)

loda.fit()

result = loda.produce(inputs=df_d3m)

result_df = result.value

result_csv_path = 'results/ETHUSDT_H1_LODA_results.csv'

result_df.to_csv(result_csv_path, index=False)
