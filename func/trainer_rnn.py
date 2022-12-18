from func.data.rnn_dataloader import RNNDataset, sort_batch, DisRNNDataset
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np
from tqdm import tqdm
#from func.models.Net_3D_conv1 import net_conv1
#from func.tools.logger import Logger
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, confusion_matrix
from func.loss.losses import LossPool
from func.models.model import model_define

import torch.nn.functional as F
from sklearn import metrics

def seed_everything(seed, cuda=True):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        self.model = model_define(self.cfig['model_name'], self.cfig[self.cfig['model_name']]).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.9, 0.999), weight_decay = 0.01)
        self.train_loader, self.val_loader, self.dict_paths = self.rnn_dataloader()

        #self.logger = Logger(osp.join(self.cfig['save_path'], 'logs'))
        self.lr = cfig['learning_rate']
        
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr   
                print ('After modify, the learning rate is', param_group['lr'])    
        
    def rnn_dataloader(self):  
        df=pd.read_csv(self.cfig['label_csv'])   
        list_IDs = list(set(df['subject'].tolist()))
        list_IDs = [str(i) for i in list_IDs]
        partition_IDs={'train':[],'validation':[], 'test':[]}
        labels, dict_paths, dict_dists = {}, {}, {}
        for i in range(len(list_IDs)):
            dict_paths[list_IDs[i]] = []
            dict_dists[list_IDs[i]] = []
        for i, item in df.iterrows():
            #path= self.cfig['mcl_path'] + '/' +item['item']
            path= self.cfig['data_path'][item['source']] + '/' +item['item']
            if not os.path.exists(path):
                print (path, 'not exist')
                continue
            sub = str(item['subject'])
#             if (item['phase'] == 'test'):
#                 partition_IDs['test'].append(sub)
#             if (item['phase'] == 'train'):
#                 partition_IDs['train'].append(sub)
#             if (item['phase'] == 'val'):
#                 partition_IDs['validation'].append(sub)
            if (item['phase'] == self.cfig['val_phase']):
                partition_IDs['validation'].append(sub)
            
            if (item['phase'] != self.cfig['val_phase']):
                partition_IDs['train'].append(sub)
                
            dict_paths[sub].append(path)
            if self.cfig['use_dis']:
                dict_dists[sub].append(item['norm_time'])
            labels[sub] = int(item['gt'])

        partition_IDs['train'] = list(set(partition_IDs['train']))
        partition_IDs['validation'] = list(set(partition_IDs['validation']))
        #partition_IDs['test'] = list(set(partition_IDs['test']))
        
        if self.cfig['add_positive']:
            print ('========================add more positive =======')
            u=[]                               
            for i in partition_IDs["train"]:
                 if(labels[i]==1): u.append(i)
            partition_IDs['train']+=u

        paramstrain = {'shuffle': True,
                  'num_workers': 8,
                  'batch_size': self.cfig['batch_size']}
        paramstest = {'shuffle': False,
                  'num_workers': 8,
                  'batch_size': self.cfig['test_batch_size']}
        sample_size = self.cfig[self.cfig['model_name']]['sample_size']
        max_step = self.cfig[self.cfig['model_name']]['max_step']
        print ('sample size is: ', sample_size)
        print ('The total training subjects is ', len(partition_IDs['train']))
        print ('The positive number is ', sum([labels[i] for i in partition_IDs['train']]))
        
        if self.cfig['use_dis']:
            training_set=DisRNNDataset(partition_IDs['train'], dict_paths, dict_dists, labels, max_step, sample_size, argu = True)
            validation_set=DisRNNDataset(partition_IDs['validation'], dict_paths, dict_dists,labels, max_step, sample_size, argu = False)
            #test_set = DisRNNDataset(partition_IDs['test'], dict_paths, dict_dists, labels, max_step, sample_size, argu = False)
        else:
            training_set=RNNDataset(partition_IDs['train'], dict_paths, labels, max_step, sample_size, argu = True)
            validation_set=RNNDataset(partition_IDs['validation'], dict_paths, labels, max_step, sample_size, argu = False)
            #test_set = RNNDataset(partition_IDs['test'], dict_paths, labels, max_step, sample_size, argu = False)
        
        
        training_generator = data.DataLoader(training_set, **paramstrain)
        validation_generator = data.DataLoader(validation_set, **paramstest)
        #test_generator = data.DataLoader(test_set, **paramstest)
        return training_generator, validation_generator, dict_paths
        
    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            if self.cfig['adjust_lr']:
                self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
                self.optim = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.9, 0.999))
            print ('learning rate: ', self.lr)
            model_root = osp.join(self.cfig['save_path'], 'models')
            #print ('learning rate: ', self.lr)
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            if os.path.exists(model_pth) and self.cfig['use_exist_model']:
                if self.device == 'cuda': #there is a GPU device
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            else:
                self.train_epoch(epoch)
                torch.save(self.model.state_dict(), model_pth)
            if self.cfig['iseval']:
                self.eval_epoch(epoch)
                #self.test_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, pos_list, target_list, loss_list = [],[],[],[]
        #print (self.dict_paths.keys())
        for batch_idx, (data, points, target, ID) in enumerate(self.train_loader):
            
            #print (ID[0], target)
#             print (len(self.dict_paths), len(self.train_loader), len(self.val_loader), len(self.test_loader))
#             print (batch_idx, '========')
            seq_len = []
            if (len(ID) != self.cfig['batch_size']):
                continue
#             for i in range(self.cfig['batch_size']):
#                 #seq_len.append(min(3, len(self.dict_paths[ID[i]])))
#                 seq_len.append(self.cfig['max_step'])                # this set just for test. time step = 3
            #time_step = min(3, len(self.dict_paths[ID[0]]))
            data = data.permute([1, 0, 2, 3, 4,5])
            if batch_idx == 0: 
                print (data.shape, 'data shape')
                print ('point example: ', points[0])
            #data, target = sort_batch(data, target, seq_len, batch_first = False)   # if use cell, we don't need here.
            #data = pack_padded_sequence(data, seq_len)               
            
            #print ('data data shape', data.data.shape)
            data, points, target = data.to(self.device), points.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            
            
            feat, pred = self.model(data)  
                        # here should be careful
            pred_prob = F.softmax(pred)

            print ('pred and target shape: ', pred.shape, target.shape)
            loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()
            try:
                print_str = 'train epoch=%d, batch_idx=%d/%d, loss=%.4f\n' % (
            epoch, batch_idx, len(self.train_loader), loss.data.item())
                if batch_idx % 5 == 0: print(print_str)
            except:
                print ('Unexpected error')
            
            pred_cls = pred.data.max(1)[1]
            pred_list += pred_cls.data.cpu().numpy().tolist()
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
        
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)     
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(train_csv):
            csv_info = ['epoch', 'lr','loss', 'auc', 'accuracy', 'f1', 'recall', 'precision']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(train_csv)
        df = pd.read_csv(train_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)
        tmp_lr = df['lr'].tolist()
        tmp_lr.append(self.lr)
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
        data['epoch'], data['loss'], data['auc'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_auc, tmp_acc
        data['f1'], data['recall'], data['precision'], data['lr'] = tmp_f1, tmp_recall, tmp_pre, tmp_lr
        data.to_csv(train_csv)
        

        
    def eval_epoch(self, epoch):  
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'eval.csv')
        pred_list, pos_list, target_list, loss_list = [],[],[], []
        with torch.no_grad():
            for batch_idx, (data, points, target, ID) in enumerate(self.val_loader):
                seq_len = []
                #print ('ID ', ID)
                if (len(ID) != self.cfig['test_batch_size']):
                    continue
    #             for i in range(self.cfig['test_batch_size']):
    #                 #seq_len.append(min(3, len(self.dict_paths[ID[i]])))
    #                 seq_len.append(self.cfig['max_step'])     # this set just for test. time step = 3

                data = data.permute([1, 0, 2, 3, 4,5])
                if batch_idx == 0: 
                    print (data.shape, 'data shape')
                    print ('point example: ', points[0])

                data, points, target = data.to(self.device), points.to(self.device), target.to(self.device)
                self.optim.zero_grad()

                feat, pred = self.model(data)  
                
                pred_prob = F.softmax(pred)
                #loss = self.criterion(pred, target)
                loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
                pred_cls = pred.data.max(1)[1]  # not test yet
                pred_list += pred_cls.data.cpu().numpy().tolist()
                pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                target_list += target.data.cpu().numpy().tolist()
                loss_list.append(loss.data.cpu().numpy().tolist())
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)     
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'auc', 'accuracy', 'f1', 'recall', 'precision']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        tmp_f1 = df['f1'].tolist()
        tmp_f1.append(f1)
        tmp_recall = df['recall'].tolist()
        tmp_recall.append(recall)
        tmp_pre = df['precision'].tolist()
        tmp_pre.append(precision)
        data['epoch'], data['loss'], data['auc'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_auc, tmp_acc
        data['f1'], data['recall'], data['precision'] = tmp_f1, tmp_recall, tmp_pre
        data.to_csv(eval_csv)
        

        
    def test_epoch(self, epoch):  
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        test_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, pos_list, target_list, loss_list = [],[],[], []
        with torch.no_grad():
            for batch_idx, (data, points, target, ID) in enumerate(self.test_loader):
                seq_len = []
                #print ('ID ', ID)
                if (len(ID) != self.cfig['test_batch_size']):
                    continue
    #             for i in range(self.cfig['test_batch_size']):
    #                 #seq_len.append(min(3, len(self.dict_paths[ID[i]])))
    #                 seq_len.append(self.cfig['max_step'])     # this set just for test. time step = 3

                data = data.permute([1, 0, 2, 3, 4,5])
                if batch_idx == 0: 
                    print (data.shape, 'data shape')
                    print ('point example: ', points[0])

                data, points, target = data.to(self.device), points.to(self.device), target.to(self.device)
                self.optim.zero_grad()

                if self.cfig['use_dis']:
                    feat, pred = self.model(data, points)  
                else:
                    feat, pred = self.model(data)             # here should be careful
                pred_prob = F.softmax(pred)
                #loss = self.criterion(pred, target)
                loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
                pred_cls = pred.data.max(1)[1]  # not test yet
                pred_list += pred_cls.data.cpu().numpy().tolist()
                pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                target_list += target.data.cpu().numpy().tolist()
                loss_list.append(loss.data.cpu().numpy().tolist())
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)     
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(test_csv):
            csv_info = ['epoch', 'loss', 'auc', 'accuracy', 'f1', 'recall', 'precision']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(test_csv)
        df = pd.read_csv(test_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        tmp_f1 = df['f1'].tolist()
        tmp_f1.append(f1)
        tmp_recall = df['recall'].tolist()
        tmp_recall.append(recall)
        tmp_pre = df['precision'].tolist()
        tmp_pre.append(precision)
        data['epoch'], data['loss'], data['auc'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_auc, tmp_acc
        data['f1'], data['recall'], data['precision'] = tmp_f1, tmp_recall, tmp_pre
        data.to_csv(test_csv)
    
    def test(self):
        model_root = osp.join(self.cfig['save_path'], 'models')
        model_list = os.listdir(model_root)
        Acc, F1, Recl, Prcn = [], [], [], []
        for epoch in range(1,85):
            print ('epoch: ', epoch)
            model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.cfig['save_path'], 'models'), epoch)
            accuracy, f1, recall, precision = self.test_epoch(model_pth)
            print (accuracy, f1, recall, precision)
            Acc.append(accuracy)
            F1.append(f1)
            Recl.append(recall)
            Prcn.append(precision)
        data = pd.DataFrame()
        data['accuracy'] = Acc
        data['f1'] = F1
        data['recall'] = Recl
        data['precision'] = Prcn
        print ('Acc: ', Acc)
        print ('f1:', F1)
        print ('Recl', Recl)
        print ('Prcn', Prcn)
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        test_csv = os.path.join(self.csv_path, 'test.csv')
        data.to_csv(test_csv)
        
