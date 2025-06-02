from tods.tods.detection_algorithm.PyodLOF import LOFPrimitive
import pandas as pd
from d3m import container

df = pd.read_csv('crypto_data/ETHUSDT_H1.csv')

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
    'n_neighbors': 20,
    'algorithm' : 'kd_tree',
    'leaf_size' : 30,
    'metric' : 'minkowski',
    'p' : 2,
    'contamination' : 0.1,
    'n_jobs' : 1
}

lof = LOFPrimitive(hyperparams=hyperparams)

lof.set_training_data(inputs=df_d3m)

lof.fit()

result = lof.produce(inputs=df_d3m)

result_df = result.value

result_csv_path = 'results/ETHUSDT_H1_LOF_results.csv'

result_df.to_csv(result_csv_path, index=False)