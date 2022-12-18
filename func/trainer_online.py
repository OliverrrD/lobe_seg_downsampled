from func.data.data_generator import CNNDataset, CNNDiag
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
from func.loss.losses import LossPool, CenterLoss
from func.models.model import model_define
import torch.nn.functional as F
from sklearn import metrics
import pdb  # debugger

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.add_reg = self.cfig[self.cfig['model_name']]['add_reg']

        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        self.model = model_define(self.cfig['model_name'], self.cfig[self.cfig['model_name']]).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.9, 0.999), weight_decay = self.cfig['weight_decay'])
        #self.train_loader, self.val_loader = self.data_loader()
        
        self.lr = cfig['learning_rate']
        
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr   
                print ('After modify, the learning rate is', param_group['lr'])
    
    def data_loader(self):  
        df=pd.read_csv(self.cfig['online_label'])   
        for source in self.cfig['exclude_source']:
            df = df.loc[df['source'] != source]
        list_IDs = list(set(df['subject'].tolist()))
        list_IDs = [str(i) for i in list_IDs]
        partition_IDs={'train':[],'validation':[], 'test':[]}
        labels, dict_paths, dict_diags = {}, {}, {}
        for i in range(len(list_IDs)):
            dict_paths[list_IDs[i]] = []
            dict_diags[list_IDs[i]] = []
        for i, item in df.iterrows():
            #path= self.cfig['data_path'] + '/' +item['item']
            path= self.cfig['data_path'][item['source']] + '/' +item['item']
            if not os.path.exists(path):
                print (path, 'not exist')
                break
            sub = str(item['subject'])
            if sub not in dict_paths.keys():
                dict_paths[sub] = []
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
            labels[sub] = int(item['gt'])
            
            if self.add_reg == True:
                dict_diags[sub].append(item['diag_log'])
#         if self.cfig['use_kaggle']:
#             df=pd.read_csv(self.cfig['kaggle_csv'])
#             for i,item in df.iterrows():
#                 path=self.cfig['kaggle_path'] + '/' + item["id"]
#                 if(os.path.exists(path+".nii.gz")):
#                     #print (item['id'])
#                     partition_IDs["train"].append(item['id'])
#                     labels[item['id']]= int(item["cancer"])
#                     dict_paths[item['id']] = [path +".nii.gz"]
        partition_IDs['train'] = list(set(partition_IDs['train']))
        partition_IDs['validation'] = list(set(partition_IDs['validation']))
        #partition_IDs['test'] = list(set(partition_IDs['test']))
        
        
#============================= select subset for test ======================================#   
        if self.cfig['small_dataset']:
           print ('use small dataset')
           partition_IDs['train'] = partition_IDs['train'][:self.cfig['small_tr']]
           partition_IDs['validation'] = partition_IDs['validation'][:self.cfig['small_val']]
        
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
        print ('sample size is: ', sample_size)
        print ('The total training subjects is ', len(partition_IDs['train']))
        print ('The positive number is ', sum([labels[i] for i in partition_IDs['train']]))
        
        if self.add_reg == False:
            training_set=CNNDataset(partition_IDs['train'], dict_paths, labels,  sample_size, argu = True, use_mask = self.cfig['use_mask'], lastscan = self.cfig['lastscan'])
            validation_set=CNNDataset(partition_IDs['validation'], dict_paths, labels, sample_size, argu = False, use_mask = self.cfig['use_mask'], lastscan = self.cfig['lastscan'])
        else:
            training_set=CNNDiag(partition_IDs['train'], dict_paths, dict_diags, labels,  sample_size, argu = True, use_mask = self.cfig['use_mask'], lastscan = self.cfig['lastscan'])
            validation_set=CNNDiag(partition_IDs['validation'], dict_paths, dict_diags, labels, sample_size, argu = False, use_mask = self.cfig['use_mask'], lastscan = self.cfig['lastscan'])
        
        print ('len of train set and val set', len(training_set), len(validation_set))
        training_generator = data.DataLoader(training_set, **paramstrain)
        validation_generator = data.DataLoader(validation_set, **paramstest)
        #test_generator = data.DataLoader(test_set, **paramstest)
        return training_generator, validation_generator#, test_generator

        
    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            self.train_loader, self.val_loader = self.data_loader()
            if self.cfig['adjust_lr']:
                self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
            print ('lr: ', self.lr)
            model_root = osp.join(self.cfig['save_path'], 'models')
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
        pred_list, target_list, loss_list, pos_list = [],[],[], []
        for batch_idx, data_tup in enumerate(self.train_loader):
            if self.add_reg == False:
                data, target, ID = data_tup
            else:
                data, diag, target, ID = data_tup
                diag = diag.float()
                diag = diag.to(self.device)
            if batch_idx == 0: print ('data shape ', data.shape, 'use diag', self.add_reg)
            data, target = data.to(self.device), target.to(self.device)
            #feat, pred = self.model(data)
            
            
            if self.add_reg == False:
                feat, pred = self.model(data)
                loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()            
            else:
                scan_t, pred = self.model(data)
                loss = LossPool(pred, target, self.cfig, scan_t = scan_t, diag_t = diag, loss_name = self.cfig['loss_name'] ).get_loss()
            pred_prob = F.softmax(pred)
            if self.cfig['add_center']:  
                centerloss = CenterLoss(2, self.cfig[self.cfig['model_name']]['dim']).cuda()

                #optimzer4center = torch.optim.SGD(centerloss.parameters(), lr =0.5)
                optimzer4center = torch.optim.Adam(self.model.parameters(), lr = 0.01, betas=(0.9, 0.999), weight_decay = self.cfig['weight_decay'])
                loss = loss + self.cfig['center_weight'] * centerloss(target, feat)
                if batch_idx % 5 == 0: 
                    print ('Center Loss:', centerloss(target, feat))
                    print ('-----------------Add Center loss-------------')
                optimzer4center.zero_grad()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print_str = 'train epoch=%d, batch_idx=%d/%d\n' % (
            epoch, batch_idx, len(self.train_loader))
            if batch_idx % 5 == 0: print(print_str)
            pred_cls = pred.data.max(1)[1]
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            pred_list += pred_cls.data.cpu().numpy().tolist()
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
            csv_info = ['epoch', 'loss', 'accuracy', 'f1', 'recall', 'precision']
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
        tmp_f1 = df['f1'].tolist()
        tmp_f1.append(f1)
        tmp_recall = df['recall'].tolist()
        tmp_recall.append(recall)
        tmp_pre = df['precision'].tolist()
        tmp_pre.append(precision)
        
        data['epoch'], data['loss'], data['accuracy'] = tmp_epoch, tmp_loss, tmp_acc
        data['f1'], data['recall'], data['precision'],  = tmp_f1, tmp_recall, tmp_pre
        data.to_csv(train_csv)
        
        #---------------------- save to tensorboard ----------------#
#         if self.cfig['save_tensorlog']:
#             self.logger.scalar_summary('loss', np.mean(loss_list), epoch + 1)
#             self.logger.scalar_summary('f1', f1, epoch + 1)
#             self.logger.scalar_summary('accuracy', accuracy, epoch + 1)
#             self.logger.scalar_summary('recall', recall, epoch + 1)
#             self.logger.scalar_summary('precision', precision, epoch + 1)

        
    def eval_epoch(self, epoch):  
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'eval.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[], []
        with torch.no_grad():
            for batch_idx, data_tup in enumerate(self.val_loader):
                if self.add_reg == False:
                    data, target, ID = data_tup
                else:
                    data, diag, target, ID = data_tup
                    diag = diag.float()
                    diag = diag.to(self.device)
                data, target = data.to(self.device), target.to(self.device)

                #feat, pred = self.model(data)             # here should be careful



                if self.add_reg == False:
                    feat, pred = self.model(data)
                    loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()            
                else:
                    scan_t, pred = self.model(data)
                    loss = LossPool(pred, target, self.cfig, scan_t = scan_t, diag_t = diag, loss_name = self.cfig['loss_name'] ).get_loss()
                pred_prob = F.softmax(pred)
                if self.cfig['add_center']:  
                    if batch_idx == 0: print ('-----------------Add Center loss-------------')
                    centerloss = CenterLoss(2, self.cfig[self.cfig['model_name']]['dim']).cuda()
                    optimzer4center = torch.optim.SGD(centerloss.parameters(), lr =0.05)
                    loss = loss + self.cfig['center_weight'] * centerloss(target, feat)
                    optimzer4center.zero_grad()
            #self.optim.zero_grad()
                pred_cls = pred.data.max(1)[1]  
                pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                pred_list += pred_cls.data.cpu().numpy().tolist()
                target_list += target.data.cpu().numpy().tolist()
                loss_list.append(loss.data.cpu().numpy().tolist())
        print (confusion_matrix(target_list, pred_list))
        print (target_list)
        print (pos_list)
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)    
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss','auc','accuracy', 'f1', 'recall', 'precision']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        print ('------------------', tmp_epoch)
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
        data['f1'], data['recall'], data['precision'] = tmp_f1, tmp_recall, tmp_pre
        data.to_csv(eval_csv)
        
            
    def test_epoch(self, epoch, phase = 'val'):        
        self.model.eval()
        model_root = osp.join(self.cfig['save_path'], 'models')
        model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
        if self.device == 'cuda': #there is a GPU device
            self.model.load_state_dict(torch.load(model_pth))
        else:
            self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        #test_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[], []
        pred_ori_list = []
        ID_list = []
        if phase == 'val':
            loader = self.val_loader
        if phase == 'train':
            loader = self.train_loader
        for batch_idx, (data, target, ID) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            
            pred = self.model(data)             # here should be careful
            pred_prob = F.softmax(pred)
            #loss = self.criterion(pred, target)
            loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            if self.cfig['add_center']:  
                print ('-----------------Add Center loss-------------')
                centerloss = CenterLoss(2, self.cfig[self.cfig['model_name']]['dim']).cuda()
                optimzer4center = torch.optim.SGD(centerloss.parameters(), lr =0.05)
                loss = loss + self.cfig['center_weight'] * centerloss(target, feat)
                optimzer4center.zero_grad()
            self.optim.zero_grad()
            pred_ori_list += pred.data.cpu().numpy().tolist()
            ID_list += ID
            pred_cls = pred.data.max(1)[1]  
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)    
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        return pred_ori_list, ID_list, target_list, roc_auc, f1, recall, precision, accuracy
#         #-------------------------save to csv -----------------------#
#         if not os.path.exists(test_csv):
#             csv_info = ['epoch', 'loss','auc','accuracy', 'f1', 'recall', 'precision']
#             init_csv = pd.DataFrame()
#             for key in csv_info:
#                 init_csv[key] = []
#             init_csv.to_csv(test_csv)
#         df = pd.read_csv(test_csv)
#         data = pd.DataFrame()
#         tmp_epoch = df['epoch'].tolist()
#         tmp_epoch.append(epoch)

#         print ('------------------', tmp_epoch)
#         tmp_loss = df['loss'].tolist()
#         tmp_loss.append(np.mean(loss_list))
#         tmp_auc = df['auc'].tolist()
#         tmp_auc.append(roc_auc)
#         tmp_acc = df['accuracy'].tolist()
#         tmp_acc.append(accuracy)
#         tmp_f1 = df['f1'].tolist()
#         tmp_f1.append(f1)
#         tmp_recall = df['recall'].tolist()
#         tmp_recall.append(recall)
#         tmp_pre = df['precision'].tolist()
#         tmp_pre.append(precision)
#         data['epoch'], data['loss'], data['auc'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_auc, tmp_acc
#         data['f1'], data['recall'], data['precision'] = tmp_f1, tmp_recall, tmp_pre
#         data.to_csv(test_csv)
        
        #---------------------- save to tensorboard ----------------#
#         if self.cfig['save_tensorlog']:
#             self.logger.scalar_summary('test_loss', np.mean(loss_list), epoch + 1)
#             self.logger.scalar_summary('test_f1', f1, epoch + 1)
#             self.logger.scalar_summary('test_auc', roc_auc, epoch + 1)
#             self.logger.scalar_summary('test_accuracy', accuracy, epoch + 1)
#             self.logger.scalar_summary('test_recall', recall, epoch + 1)
#             self.logger.scalar_summary('test_precision', precision, epoch + 1)
  
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
        

