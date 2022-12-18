from func.data.data_generator import CNNSurv
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np
from tqdm import tqdm
# from func.models.Net_3D_conv1 import net_conv1
# from func.tools.logger import Logger
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from func.loss.losses import LossPool, CenterLoss
from func.models.model import model_define
import torch.nn.functional as F
from sklearn import metrics
import pdb  # debugger


class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.add_reg = self.cfig['model_params']['add_reg']
        self.disbce = self.cfig['loss_name'][:6] == 'disbce'
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        self.model = model_define(self.cfig['model_name'], self.cfig['model_params']).to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfig['learning_rate'], betas=(0.9, 0.999),
                                      weight_decay=self.cfig['weight_decay'])
        self.train_loader, self.val_loader, self.test_loader = self.data_loader()

        self.lr = cfig['learning_rate']

    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
                print('After modify, the learning rate is', param_group['lr'])

    def data_loader(self):
        df = pd.read_csv(self.cfig['label_csv'])
        df = df.loc[df['is_exist'] == 1]
        need_factor = ['age', 'edu', 'bmi', 'copd', 'phist', 'fhist', 'sstatus', 'sintensity', 'dur', 'quit', 'race']
        for source in self.cfig['exclude_source']:
            df = df.loc[df['source'] != source]
        list_IDs = list(set(df['subject'].tolist()))
        list_IDs = [str(i) for i in list_IDs]
        # print (len(list_IDs))
        partition_IDs = {'train': [], 'validation': [], 'test': []}
        labels, dict_paths, dict_diags = {}, {}, {}
        path_labels = {}
        path_factors = {}
        for i in range(len(list_IDs)):
            dict_paths[list_IDs[i]] = []
            dict_diags[list_IDs[i]] = []
        for i, item in df.iterrows():
            tmp_factors = []
            for name in need_factor:
                tmp_factors.append(item[name])


            path = self.cfig['data_path'][item['source']] + '/' + item['item']
            path_factors[path] = tmp_factors

            sub = str(item['subject'])
            if sub not in dict_paths.keys():
                dict_paths[sub] = []

            if (item['phase'] == self.cfig['val_phase']):
                partition_IDs['validation'].append(sub)

            if (item['phase'] == self.cfig['tt_phase']):
                partition_IDs['test'].append(sub)

            if (item['phase'] != self.cfig['val_phase'] and item['phase'] != self.cfig['tt_phase']):
                partition_IDs['train'].append(sub)
            dict_paths[sub].append(path)
            labels[sub] = int(item['gt_reg'])
            if self.cfig['gt_reg'] == True:
                path_labels[path] = int(item['gt_reg'])
            else:
                path_labels[path] = int(item['gt'])

            if self.add_reg or self.disbce:
                dict_diags[sub].append(item['diag_dis'])

        partition_IDs['train'] = list(set(partition_IDs['train']))
        partition_IDs['validation'] = list(set(partition_IDs['validation']))
        partition_IDs['test'] = list(set(partition_IDs['test']))
        print('train, val, test samples is: ', len(partition_IDs['train']), len(partition_IDs['validation']),
              len(partition_IDs['test']))
        assert (len(set(partition_IDs['test']) & set(partition_IDs['validation'])) == 0)
        assert (len(set(partition_IDs['train']) & set(partition_IDs['validation'])) == 0)
        assert (len(set(partition_IDs['test']) & set(partition_IDs['train'])) == 0)

        if self.cfig['add_positive']:
            print('========================add more positive =======')
            u = []
            for i in partition_IDs["train"]:
                if (labels[i] == 1): u.append(i)
            partition_IDs['train'] += u
        if self.cfig['add_negative']:
            print('=======================add more negative ========')
            u = []
            for i in partition_IDs["train"]:
                if (labels[i] == 0): u.append(i)
            partition_IDs['train'] += u

        paramstrain = {'shuffle': True,
                       'num_workers': 4,
                       'batch_size': self.cfig['batch_size']}
        paramstest = {'shuffle': False,
                      'num_workers': 4,
                      'batch_size': self.cfig['test_batch_size']}
        sample_size = self.cfig['sample_size']
        print('sample size is: ', sample_size)
        print('The total training subjects is ', len(partition_IDs['train']))
        print('The positive subject in training is ', sum([labels[i] for i in partition_IDs['train']]))
        sum_pos = 0
        for subj in partition_IDs['train']:
            paths = dict_paths[subj]
            for path in paths:
                sum_pos += path_labels[path]
        print('The total positive item in training is: ', sum_pos)


        training_set = CNNSurv(partition_IDs['train'], dict_paths, path_factors, path_labels, sample_size, argu=True,
                                      use_mask=self.cfig['use_mask'], lastscan=self.cfig['lastscan'])
        validation_set = CNNSurv(partition_IDs['validation'], dict_paths, path_factors, path_labels, sample_size, argu=False,
                                        use_mask=self.cfig['use_mask'], lastscan=self.cfig['lastscan'])
        test_set = CNNSurv(partition_IDs['test'], dict_paths, path_factors, path_labels, sample_size, argu=False,
                                  use_mask=self.cfig['use_mask'], lastscan=self.cfig['lastscan'])



        print('len of train set and val set', len(training_set), len(validation_set))
        training_generator = data.DataLoader(training_set, **paramstrain)
        validation_generator = data.DataLoader(validation_set, **paramstest)
        test_generator = data.DataLoader(test_set, **paramstest)
        return training_generator, validation_generator, test_generator

    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            if self.cfig['adjust_lr']:
                self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
            print('lr: ', self.lr)
            model_root = osp.join(self.cfig['save_path'], 'models')
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            if os.path.exists(model_pth) and self.cfig['use_exist_model']:
                if self.device == 'cuda':  # there is a GPU device
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            else:

                self.train_epoch(epoch)
                torch.save(self.model.state_dict(), model_pth)
            if self.cfig['iseval']:
                self.eval_epoch(epoch, 'eval')
                self.eval_epoch(epoch, 'test')

    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list, pos_list = [], [], [], []
        regloss_list = []
        for batch_idx, data_tup in enumerate(self.train_loader):

            data, factors, target,  ID = data_tup


            if batch_idx == 0: print('data shape ', data.shape, 'use diag', self.add_reg or self.disbce)

            data, target = data.to(self.device), target.to(self.device)
            factors = factors.to(self.device)
            # feat, pred = self.model(data)
            print('The loss function using is: ', self.cfig['loss_name'])
            feat, pred = self.model(data, factors)
            print(pred.shape, 'pred.shape')
            loss0 = LossPool(feat, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            loss1 = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            loss = loss0 + 0.2 * loss1


            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print_str = 'train epoch=%d, batch_idx=%d/%d\n' % (
                epoch, batch_idx, len(self.train_loader))
            if batch_idx % 5 == 0: print(print_str)
            if len(pred.shape) == 2:
                pred_prob = F.softmax(pred, dim=1)
                pred_cls = pred.data.max(1)[1]
                pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                pred_list += pred_cls.data.cpu().numpy().tolist()
            else:
                # print (pred, target)
                pos_list += pred.data.cpu().numpy().tolist()
                pred_list += (pred > 0.5).tolist()
            target_list += target.data.cpu().numpy().tolist()
            #loss_list.append(loss.data.cpu().numpy().tolist())
            loss_list.append([loss0.data.cpu().numpy().tolist(), loss1.data.cpu().numpy().tolist()])

            # break
        print(confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)

        roc_auc = metrics.auc(fpr, tpr)
        f1 = f1_score(target_list, pred_list)
        recall = recall_score(target_list, pred_list)
        precision = precision_score(target_list, pred_list)
        accuracy = accuracy_score(target_list, pred_list)
        # -------------------------save to csv -----------------------#
        if not os.path.exists(train_csv):
            csv_info = ['epoch', 'loss', 'accuracy', 'auc', 'f1', 'recall', 'precision']
            if self.add_reg: csv_info.append('regloss')
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(train_csv)
        df = pd.read_csv(train_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        print('------------------', tmp_epoch)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        tmp_f1 = df['f1'].tolist()
        tmp_f1.append(f1)
        tmp_recall = df['recall'].tolist()
        tmp_recall.append(recall)
        tmp_pre = df['precision'].tolist()
        tmp_pre.append(precision)

        data['epoch'], data['loss'], data['accuracy'] = tmp_epoch, tmp_loss, tmp_acc
        data['auc'] = tmp_auc
        data['f1'], data['recall'], data['precision'] = tmp_f1, tmp_recall, tmp_pre
        if self.add_reg:
            tmp_regloss = df['regloss'].tolist()
            tmp_regloss.append(np.mean(regloss_list))
            data['regloss'] = tmp_regloss
        data.to_csv(train_csv)

    def eval_epoch(self, epoch, phase):
        # model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        # self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, phase + '.csv')
        pred_list, target_list, loss_list, pos_list = [], [], [], []
        regloss_list = []
        if phase == 'eval':
            loader = self.val_loader
        if phase == 'test':
            loader = self.test_loader
        with torch.no_grad():
            for batch_idx, data_tup in enumerate(loader):

                data, factors, target, ID = data_tup

                data, target = data.to(self.device), target.to(self.device)
                factors = factors.to(self.device)


                feat, pred = self.model(data, factors)
                loss0 = LossPool(feat, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()
                loss1 = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()


                self.optim.zero_grad()
                if len(pred.shape) == 2:
                    pred_prob = F.softmax(pred, dim=1)
                    pred_cls = pred.data.max(1)[1]
                    pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                    pred_list += pred_cls.data.cpu().numpy().tolist()
                else:
                    # print (pred, target)
                    pos_list += pred.data.cpu().numpy().tolist()
                    pred_list += (pred > 0.5).tolist()
                target_list += target.data.cpu().numpy().tolist()
                #loss_list.append(loss.data.cpu().numpy().tolist())
                loss_list.append([loss0.data.cpu().numpy().tolist(), loss1.data.cpu().numpy().tolist()])

        print(confusion_matrix(target_list, pred_list))
        print(target_list)
        print(pos_list)
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)
        f1 = f1_score(target_list, pred_list)
        recall = recall_score(target_list, pred_list)
        precision = precision_score(target_list, pred_list)
        accuracy = accuracy_score(target_list, pred_list)
        # -------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'auc', 'accuracy', 'f1', 'recall', 'precision']
            if self.add_reg: csv_info.append('regloss')
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        print('------------------', tmp_epoch)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        tmp_f1 = df['f1'].tolist()
        tmp_f1.append(f1)
        tmp_recall = df['recall'].tolist()
        tmp_recall.append(recall)
        tmp_pre = df['precision'].tolist()
        tmp_pre.append(precision)
        data['epoch'], data['loss'], data['auc'], data['accuracy'] = tmp_epoch, tmp_loss, tmp_auc, tmp_acc
        data['f1'], data['recall'], data['precision'] = tmp_f1, tmp_recall, tmp_pre

        if self.add_reg:
            tmp_regloss = df['regloss'].tolist()
            tmp_regloss.append(np.mean(regloss_list))
            data['regloss'] = tmp_regloss
        data.to_csv(eval_csv)

    def test_epoch(self, epoch, phase='val'):
        self.model.eval()
        model_root = osp.join(self.cfig['save_path'], 'models')
        model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
        if self.device == 'cuda':  # there is a GPU device
            self.model.load_state_dict(torch.load(model_pth))
        else:
            self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))

        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        # test_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, target_list, loss_list, pos_list = [], [], [], []
        regloss_list, gt_reg = [], []
        ID_list, scan_t_list, diag_list = [], [], []
        if phase == 'val':
            loader = self.val_loader
        if phase == 'train':
            loader = self.train_loader
        if phase == 'test':
            loader = self.test_loader
        with torch.no_grad():
            for batch_idx, data_tup in enumerate(loader):
                print(batch_idx)
                if self.add_reg == False and not self.disbce:
                    data, target, ID = data_tup
                else:
                    data, diag, target, y_reg, ID = data_tup
                    diag = diag.float()
                    diag = diag.to(self.device)
                    y_reg = y_reg.to(self.device)
                data, target = data.to(self.device), target.to(self.device)

                if self.disbce == True:
                    feat, pred = self.model(data)
                    loss = LossPool(pred, target.float(), self.cfig, diag_t=diag,
                                    loss_name=self.cfig['loss_name']).get_loss()
                else:
                    if self.add_reg == False:
                        feat, pred = self.model(data)
                        loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
                    else:
                        scan_t, pred = self.model(data)
                        assert scan_t.shape[-1] == 1
                        scan_t = scan_t[:, 0]
                        assert scan_t.shape == diag.shape
                        assert self.cfig['loss_name'] == 'reg_bl'
                        regloss = LossPool(pred, y_reg, self.cfig, scan_t=scan_t, diag_t=diag,
                                           loss_name=self.cfig['loss_name']).get_loss()
                        celoss = LossPool(pred, target, self.cfig, loss_name='cross_entropy_loss').get_loss()
                        loss = self.cfig['alpha'] * regloss + celoss
                        print('reg loss: ', regloss.data.cpu().numpy(), 'celoss: ', celoss.data.cpu().numpy(),
                              'total loss', loss.data.cpu().numpy())
                    pred_prob = F.softmax(pred, dim=1)

                self.optim.zero_grad()
                ID_list += ID
                if len(pred.shape) == 2:
                    pred_cls = pred.data.max(1)[1]
                    pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                    pred_list += pred_cls.data.cpu().numpy().tolist()
                else:
                    # print (pred, target)
                    pos_list += pred.data.cpu().numpy().tolist()
                    pred_list += (pred > 0.5).tolist()
                target_list += target.data.cpu().numpy().tolist()
                loss_list.append(loss.data.cpu().numpy().tolist())
                if self.add_reg:
                    regloss_list.append(regloss.data.cpu().numpy().tolist())
                    diag_list += diag.data.cpu().numpy().tolist()
                    scan_t_list += scan_t.data.cpu().numpy().tolist()
                    gt_reg += y_reg.data.cpu().numpy().tolist()

        print(confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)
        f1 = f1_score(target_list, pred_list)
        recall = recall_score(target_list, pred_list)
        precision = precision_score(target_list, pred_list)
        accuracy = accuracy_score(target_list, pred_list)
        if not self.add_reg:
            gt_reg = [0] * len(pos_list)
            diag_list = [0] * len(pos_list)
            scan_t_list = [0] * len(pos_list)
        return gt_reg, pos_list, diag_list, ID_list, scan_t_list, target_list, roc_auc, f1, recall, precision, accuracy

