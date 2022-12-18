from func.data.data_generator import GetLoader
import os
import os.path as osp
import torch

from torch.utils import data
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
from func.loss.losses import LossPool

from func.models.model import model_define
import torch.nn.functional as F
from sklearn import metrics


class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print ('use gpu? ', self.device)
        self.cfig = cfig
        self.need_diag = self.cfig['need_diag']
        self.add_reg = self.cfig['model_params']['add_reg']
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        self.model = model_define(self.cfig['model_name'], self.cfig['model_params']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.9, 0.999), weight_decay = self.cfig['weight_decay'])
        self.train_loader, self.val_loader, self.test_loader, self.ext_loader = self.data_loader()

        #self.logger = Logger(osp.join(self.cfig['save_path'], 'logs'))

        self.lr = cfig['learning_rate']
        
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr   
                print ('After modify, the learning rate is', param_group['lr'])
        

    def data_loader(self):
        df=pd.read_csv(self.cfig['label_csv'])
        # df = df.loc[df['is_exist'] == 1]
        # if self.cfig['balance']:
        #     df = df[df['balance'] == 1]

        for source in self.cfig['exclude_source']:
            df = df.loc[df['source'] != source]
        for i in self.cfig['exclude_phase']:
            df = df.loc[df['phase'] != i]

        list_IDs = list(set(df['subject'].tolist()))
        list_IDs = [str(i) for i in list_IDs]
        partition_IDs={'train':[],'validation':[], 'test':[]}
        labels, dict_paths, dict_diags = {}, {}, {}
        path_labels = {}
        for i in range(len(list_IDs)):
            dict_paths[list_IDs[i]] = []
            dict_diags[list_IDs[i]] = []
        for i, item in df.iterrows():
            #print (item['source'])
            path= self.cfig['data_path'][item['source']] + '/' +item['item']
            if not os.path.exists(path):
                print (path, 'not exist')
                continue

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
            path_labels[path] = int(item['gt'])
            
            if self.need_diag:
                dict_diags[sub].append(item['diag_dis'])

        if self.cfig['external_test']:
            df = pd.read_csv(self.cfig['ext_csv'])
            list_IDs = list(set(df['subject'].tolist()))
            list_IDs = [str(i) for i in list_IDs]
            ext_ids = []

            for i in range(len(list_IDs)):
                dict_paths[list_IDs[i]] = []
                dict_diags[list_IDs[i]] = []
            for i, item in df.iterrows():

                path = self.cfig['data_path'][item['source']] + '/' + item['item']
                if not os.path.exists(path):
                    print(path, 'not exist')
                    continue

                sub = str(item['subject'])
                if sub not in dict_paths.keys():
                    dict_paths[sub] = []
                ext_ids.append(sub)
                dict_paths[sub].append(path)
                labels[sub] = int(item['gt_reg'])
                path_labels[path] = int(item['gt'])
                if self.need_diag:
                    dict_diags[sub].append(item['diag_dis'])
        

        partition_IDs['train'] = list(set(partition_IDs['train']))
        partition_IDs['validation'] = list(set(partition_IDs['validation']))
        partition_IDs['test'] = list(set(partition_IDs['test']))

        ext_ids = list(set(ext_ids))

        print ('train, val, test samples is: ', len(partition_IDs['train']), len(partition_IDs['validation']), len(partition_IDs['test']))
        assert (len(set(partition_IDs['test']) & set(partition_IDs['validation'])) == 0)
        assert (len(set(partition_IDs['train']) & set(partition_IDs['validation'])) == 0)
        print (set(partition_IDs['test']) & set(partition_IDs['train']))
        assert (len(set(partition_IDs['test']) & set(partition_IDs['train'])) == 0)
        
        
        if self.cfig['add_positive']:
            print ('========================add more positive =======')
            u=[]                               
            for i in partition_IDs["train"]:
                 if(labels[i]==1): u.append(i)
            partition_IDs['train']+=u
        if self.cfig['add_special']:
            print ("=============add some special things =============")
            print ("the length of partition_IDs['train']", len(partition_IDs['train']))
            u = []
            spore_list = df.loc[df['source'] == 'spore']['subject'].tolist()
            #print (partition_IDs['train'], spore_list)
            for id in spore_list:
                if str(id) in partition_IDs["train"]:
                    u.append(str(id))
            partition_IDs['train'] += u
            print("the length of partition_IDs['train']", len(partition_IDs['train']))


        paramstrain = {'shuffle': True,
                  'num_workers': 4,
                  'batch_size': self.cfig['batch_size']}
        paramstest = {'shuffle': False,
                  'num_workers': 4,
                  'batch_size': self.cfig['test_batch_size']}
        sample_size = self.cfig['loader_params']['sample_size']
#        global_size = 128 #self.cfig[self.cfig['model_name']]['global_size']
#        nod_size = 64
        #num_slice = self.cfig[self.cfig['model_name']]['num_slice']
        num_patch = self.cfig['loader_params']['num_patch'] 


        print ('sample size is: ', sample_size)
        print ('The total training subjects is ', len(partition_IDs['train']))
        print ('The positive number is ', sum([labels[i] for i in partition_IDs['train']]))
        sum_pos = 0
        for subj in partition_IDs['train']:
            paths = dict_paths[subj]
            for path in paths:
                sum_pos += path_labels[path]
        print ('The total positive item in training is: ', sum_pos)
        loaderfig = self.cfig['loader_params']
        loaderfig['dict_paths'] = dict_paths
        loaderfig['path_labels'] = path_labels
        loaderfig['loader_name'] = self.cfig['patch_loader']
        if self.need_diag:
            loaderfig['dict_diags'] = dict_diags
            loaderfig['labels'] = labels
        
        training_set = GetLoader(list_IDs= partition_IDs['train'], argu = self.cfig['argu'], argu_3d = self.cfig['argu_3d'], **loaderfig).get_loader()
        validation_set = GetLoader(list_IDs= partition_IDs['validation'], argu = False,argu_3d = False, **loaderfig).get_loader()
        test_set = GetLoader(list_IDs = partition_IDs['test'], argu = False,argu_3d = False, **loaderfig).get_loader()
        if self.cfig['external_test']:
            ext_set= GetLoader(list_IDs = ext_ids, argu = False,argu_3d = False, **loaderfig).get_loader()

        training_generator = data.DataLoader(training_set, **paramstrain)
        validation_generator = data.DataLoader(validation_set, **paramstest)
        test_generator = data.DataLoader(test_set, **paramstest)
        if self.cfig['external_test']:
            ext_generator = data.DataLoader(ext_set, **paramstest)
        else:
            ext_generator = None

        return training_generator, validation_generator, test_generator, ext_generator

        
        
    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            if self.cfig['adjust_lr']:
                self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
            print ('Learning rate: ', self.lr)
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
                if self.cfig['save_model']:
                    torch.save(self.model.state_dict(), model_pth)
            if self.cfig['iseval']:
                if self.cfig['external_test']:
                    self.eval_epoch(epoch, 'ext')
                self.eval_epoch(epoch, 'eval')
                self.eval_epoch(epoch, 'test')
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[],[]
        print ('The length of val and train loader: ', len(self.val_loader), len(self.train_loader))
        for batch_idx, data_tup in enumerate(self.train_loader):
            
            if self.add_reg or self.need_diag:
                data, diag, target, y_reg, ID = data_tup
                diag = diag.float()
                diag = diag.to(self.device)
                y_reg = y_reg.to(self.device)
                
            else:
                data, target, ID = data_tup
            data, target = data.to(self.device), target.to(self.device)
            # if batch_idx % 5 == 0:
            #     for i in range(len(data)):
            #         nii_img = nib.Nifti1Image(data[i].data.cpu().numpy(), affine = np.eye(4))
            #         nib.save(nii_img, '/nfs/masi/gaor2/tmp/train_nifti/ori/' + ID[i] + '.nii.gz')
                    
            self.optim.zero_grad()
            feat, pred = self.model(data)             # here should be careful
            #pred = self.model(data) 
                
            if batch_idx == 0:
                try:
                    print ('data.shape: ', data.shape)
                    print ('pred.shape, target.shape: ', pred.shape, target.shape)
                except:
                    print ('data.shape:', data[0].shape, data[1].shape)
                    print ('pred.shape, target.shape: ', pred.shape, target.shape)
            print (pred.shape, target.shape)
            if self.need_diag == True:
                if self.cfig['loss_name'] == 'reg_bl':
                    regloss = regloss = LossPool(pred, y_reg, self.cfig, scan_t = feat, diag_t = diag, loss_name = self.cfig['loss_name']).get_loss()
                    celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()
                    loss = self.cfig['alpha'] * regloss + celoss
                    print('reg loss: ', regloss.data.cpu().numpy(), 'celoss: ', celoss.data.cpu().numpy(), 'total loss',
                          loss.data.cpu().numpy())
                else:
                    loss = LossPool(pred, target.float(),  self.cfig,diag_t = diag, loss_name=self.cfig['loss_name']).get_loss()
            else:
                loss = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()

            
            loss.backward()
            self.optim.step()
            #print (loss.data)
            print_str = 'train epoch=%d, batch_idx=%d/%d, loss = %f\n' % (
            epoch, batch_idx, len(self.train_loader), loss.data.cpu().numpy())
            if batch_idx % 5 == 1: print(print_str)
            if len(pred.shape) == 2:
                pred_prob = F.softmax(pred, dim = 1)
                pred_cls = pred.data.max(1)[1]
                pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                pred_list += pred_cls.data.cpu().numpy().tolist()
            else:
                #print (pred, target)
                pos_list += pred.data.cpu().numpy().tolist()
                pred_list += (pred>0.5).tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
        print ('the training confusion matrix: ')
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr) 
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(train_csv):
            csv_info = ['epoch', 'auc','loss', 'accuracy', 'f1', 'recall', 'precision']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(train_csv)
        df = pd.read_csv(train_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print('------------------', tmp_epoch)
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
        data.to_csv(train_csv)
        
    def eval_epoch(self, epoch, phase):  
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, phase + '.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[], []
        regloss_list = []
        if phase == 'eval':
            loader = self.val_loader
        if phase == 'test':
            loader = self.test_loader

        if phase == 'ext':
            loader = self.ext_loader
        with torch.no_grad():
            for batch_idx, data_tup in enumerate(loader):
                if self.add_reg or self.need_diag:
                    
                    data, diag, target, y_reg, ID = data_tup
                    diag = diag.float()
                    diag = diag.to(self.device)
                    y_reg = y_reg.to(self.device)
                else:
                    data, target, ID = data_tup
                
                data, target = data.to(self.device), target.to(self.device)
                
                self.optim.zero_grad()
                feat, pred = self.model(data)             # here should be careful

                if self.need_diag == True:
                    if self.cfig['loss_name'] == 'reg_bl':
                        regloss = LossPool(pred, y_reg, self.cfig, scan_t=feat, diag_t=diag,
                                                     loss_name=self.cfig['loss_name']).get_loss()
                        celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()
                        loss = self.cfig['alpha'] * regloss + celoss
                        print('reg loss: ', regloss.data.cpu().numpy(), 'celoss: ', celoss.data.cpu().numpy(),
                              'total loss',
                              loss.data.cpu().numpy())
                    else:
                        loss = LossPool(pred, target.float(), self.cfig, diag_t=diag,
                                        loss_name=self.cfig['loss_name']).get_loss()
                else:
                    loss = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()

                if len(pred.shape) == 2:
                    pred_prob = F.softmax(pred, dim = 1)
                    pred_cls = pred.data.max(1)[1]
                    pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                    pred_list += pred_cls.data.cpu().numpy().tolist()
                else:
                    pos_list += pred.data.cpu().numpy().tolist()
                    pred_list += (pred>0.5).tolist()
                target_list += target.data.cpu().numpy().tolist()
                loss_list.append(loss.data.cpu().numpy().tolist())
                print ('the validation loss is: %f'% loss.data.cpu().numpy())

        print ('\nthe validation confusion matrix: ')
        print (confusion_matrix(target_list, pred_list))
        
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr) 
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'auc','accuracy', 'f1', 'recall', 'precision']
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
        def swap(a, index1, index2):
            tmp = a[:, index1: index1 + 1]
            a[:, 1 * index1: 1 * index1 + 1] = a[:,  index2:  index2 +1]
            a[:,  1* index2: 1* index2  + 1] = tmp
            return a
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
        regloss_list, gt_reg =  [], []
        ID_list, scan_t_list, diag_list = [], [], []
        if phase == 'val':
            loader = self.val_loader
        if phase == 'train':
            loader = self.train_loader
        if phase == 'test':
            loader = self.test_loader
        print ('length of loader: ', len(loader))
        with torch.no_grad():
            for batch_idx, data_tup in enumerate(loader):
                if self.add_reg or self.need_diag:

                    data, diag, target, y_reg, ID = data_tup
                    diag = diag.float()
                    diag = diag.to(self.device)
                    y_reg = y_reg.to(self.device)
                else:
                    data, target, ID = data_tup
                #data = swap(data, 1, 2)
                data, target = data.to(self.device), target.to(self.device)

                self.optim.zero_grad()
                feat, pred = self.model(data)             # here should be careful

 #               loss = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()

                if len(pred.shape) == 2:
                    pred_prob = F.softmax(pred, dim = 1)
                    pred_cls = pred.data.max(1)[1]
                    pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                    pred_list += pred_cls.data.cpu().numpy().tolist()
                else:
                    pos_list += pred.data.cpu().numpy().tolist()
                    pred_list += (pred>0.5).tolist()
                target_list += target.data.cpu().numpy().tolist()
 #               loss_list.append(loss.data.cpu().numpy().tolist())
                ID_list += ID
  #              print ('the validation loss is: %f'% loss.data.cpu().numpy())

        print ('\nthe validation confusion matrix: ')
        print (confusion_matrix(target_list, pred_list))
            
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)    
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        if not self.add_reg: 
            gt_reg = [0] * len(pos_list)
            diag_list = [0] * len(pos_list)
            scan_t_list = [0] * len(pos_list)
        return gt_reg, pos_list, diag_list, ID_list, scan_t_list, target_list, roc_auc, f1, recall, precision, accuracy

    
    def test_scan(self, epoch):
        from torch.utils import data
        self.model.eval()
        model_root = osp.join(self.cfig['save_path'], 'models')
        model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
        if self.device == 'cuda': #there is a GPU device
            self.model.load_state_dict(torch.load(model_pth))
        else:
            self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
        test_set = GetLoader(list_IDs = [], loader_name = 'singlescan', data_path = '/nfs/masi/gaor2/data/MCL/MCLnorm/img/213094435time20080207.nii.gz').get_loader()
        paramstest = {'shuffle': False,
                  'num_workers': 4,
                  'batch_size': self.cfig['test_batch_size']}
        test_generator = data.DataLoader(test_set, **paramstest)
        pos_list, coords_x, coords_y, coords_z = [], [], [], []
        with torch.no_grad():
            for batch_idx, (data, coords) in enumerate(test_generator):
                if (batch_idx % 5 == 0): print (batch_idx)
                data = data.to(self.device)
                self.optim.zero_grad()
                feat, pred = self.model(data)             # here should be careful

                if len(pred.shape) == 2:
                    pred_prob = F.softmax(pred, dim = 1)
                    pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
                else:
                    pos_list += pred.data.cpu().numpy().tolist()
                coords_x += coords[0].tolist()
                coords_y += coords[1].tolist()
                coords_z += coords[2].tolist()
                #print (coords)
                #if batch_idx > 1: break
                
        #print (len(pos_list))        
        df = pd.DataFrame()
        df['prob'] = pos_list
        df['coords_x'] = coords_x
        df['coords_y'] = coords_y
        df['coords_z'] = coords_z
        df.to_csv('/nfs/masi/gaor2/saved_file/PatchLung/2set/trueswap/val0/singlescan.csv', index = False)
        
        
