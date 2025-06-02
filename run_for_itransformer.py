import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import Namespace
from exp.exp_classification import Exp_Classification
from Mydataset.data_load import MyDataset
import torch

# Args setup for iTransformer
args = Namespace(**{
    'seq_len': 12,
    'label_len': 48,
    'pred_len': 48,
    'd_model': 256,
    'n_heads': 8,
    'e_layers': 3,
    'd_layers': 1,
    'd_ff': 2048,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,
    'train_epochs': 50,
    'gpu_type': 'cuda',
    'task_name': 'classification',
    'model_id': 'iTransformer',
    'model': 'iTransformer',   
    'data': 'MyDataset',
    'features': 'S',
    'use_gpu': True,
    'use_multi_gpu': False,
    'gpu': 0,
    'devices': '0',
    'root_path': '/home/murat/tods/Time-Series-Library/dataset/MyDataset/',
    'des': 'MyExp',
    'itr': 1,
    'embed': 'timeF',
    'distil': True,
    'factor': 5,
    'd_conv': 4,
    'dropout': 0.1,
    'freq': 'h',
    'num_workers': 4,
    'enc_in': 5,     
    'dec_in': 5,
    'activation': 'gelu',
    'c_out': 1,      
    'patience': 10,
    'checkpoints': '/home/murat/tods/Time-Series-Library/Checkpoints1/',
    'moving_avg': 25
})

# Initialize experiment
exp = Exp_Classification(args)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    MyDataset(args),
    batch_size=args.batch_size,
    shuffle=True
)

# Run training and testing
exp_name = 'iTransformer'
#exp.train(exp_name)
exp.test(exp_name,test=1)
