from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if len(batch) == 4:
                    batch_x, label, padding_mask, x_mark_enc = batch
                elif len(batch) == 3:
                    batch_x, label, padding_mask = batch
                    x_mark_enc = None
                else:
                    raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
            
                pred = outputs.detach().cpu()
                label = label.view(-1).long().cpu()  # Ensures shape is [batch_size]
                loss = criterion(pred, label)
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample

        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                if len(batch) == 4:
                    batch_x, label, padding_mask, x_mark_enc = batch
                elif len(batch) == 3:
                    batch_x, label, padding_mask = batch
                    x_mark_enc = None
                else:
                    raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                if self.args.model == 'Autoformer' :
                    outputs = self.model(batch_x, padding_mask, x_mark_enc, None)
                elif self.args.model =='Informer': 
                    outputs = self.model(batch_x, padding_mask, x_mark_enc, None)
                elif self.args.model =='iTransformer': 
                    outputs = self.model(batch_x, padding_mask, x_mark_enc, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                
                logits = outputs
                label = label.view(-1).long()
                if label.size(0) == logits.size(0):
                    loss = criterion(logits, label)
                else:
                    print(f"Skipping batch due to size mismatch: outputs {outputs.size(0)}, label {label.size(0)}")
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting, test=0):


        # Get test data and loader
        test_data, test_loader = self._get_data(flag='TEST')

        # Load model checkpoint if testing
        if test:
            print('Loading model...')
            checkpoint_path = os.path.join('/home/murat/tods/Time-Series-Library/Checkpoints_FedFormer', setting, 'checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Error: Checkpoint not found at {checkpoint_path}")
                return
            self.model.load_state_dict(torch.load(checkpoint_path))

        # Set model to evaluation mode
        self.model.eval()

        all_logits = []
        all_labels = []

        # Inference loop
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(batch) == 4:
                    batch_x, batch_y, padding_mask, x_mark_enc = batch
                elif len(batch) == 3:
                    batch_x, batch_y, padding_mask = batch
                    x_mark_enc = None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                if x_mark_enc is not None:
                    x_mark_enc = x_mark_enc.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)  # or include x_mark_enc if your model expects it
                all_logits.append(outputs.detach().cpu())
                all_labels.append(batch_y.detach().cpu())

        # Stack all predictions and true labels
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy().flatten()

        # Get probabilities & binary predictions
        probs = F.softmax(all_logits, dim=1).numpy()
        preds_binary = (probs[:, 1] > 0.5).astype(int)

        # Compute metrics
        accuracy = cal_accuracy(preds_binary, all_labels)
        precision = precision_score(all_labels, preds_binary, zero_division=0)
        recall = recall_score(all_labels, preds_binary, zero_division=0)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)

        print(f'accuracy: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1 score: {f1}')

        # Save metrics
        folder_path = f'./results/{setting}/'
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'result_classification.txt'), 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'accuracy: {accuracy}\n')
            f.write(f'precision: {precision}\n')
            f.write(f'recall: {recall}\n')
            f.write(f'f1 score: {f1}\n\n')

        # Save predictions and labels for visualization
        np.save(os.path.join(folder_path, f'{setting}_probs.npy'), probs[:, 1])  # probability of class 1
        np.save(os.path.join(folder_path, f'{setting}_trues.npy'), all_labels)

        return

            

