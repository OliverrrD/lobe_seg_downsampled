from func.data.rnn_dataloader import RNNDataset, sort_batch, RnnPatch_loader, RnnPatch_loader2, DisRnnPatch_loader, RNNMultiS_loader, RNNDiagMultiS_loader,RNNDiagMultiS_REG_loader
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
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
from func.loss.losses import LossPool
from func.models.model import model_define
import torch.nn.functional as F
from sklearn import metrics
import nibabel as nib

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        self.model = model_define(self.cfig['model_name'], self.cfig[self.cfig['model_name']]).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.9, 0.999), weight_decay = self.cfig['weight_decay'])
        self.train_loader, self.val_loader, self.test_loader, self.ext_loader = self.data_loader()

        
        self.lr = cfig['learning_rate']
        
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr   
                print ('After modify, the learning rate is', param_group['lr'])
        
    def data_loader(self):  
        df=pd.read_csv(self.cfig['label_csv'])   
        list_IDs = list(set(df['subject'].tolist()))   # ID represents subject ID
        list_IDs = [str(i) for i in list_IDs]          # transfer all the ID to string
        partition_IDs={'train':[],'validation':[], 'test':[]}
        labels, dict_paths,  = {}, {}   # dict_dists, {}
        dict_diags, dict_dis = {}, {}
        path_labels = {}
        for i in range(len(list_IDs)):
            dict_paths[list_IDs[i]] = []
            #dict_dists[list_IDs[i]] = [] # this is the old version
            dict_dis[list_IDs[i]] = []
            dict_diags[list_IDs[i]] = []
        for i, item in df.iterrows():
            path= self.cfig['data_path'][item['source']] + '/' +item['item']
            if not os.path.exists(path):
                print (path, 'not exist')
                continue
            sub = str(item['subject'])
            if sub not in dict_paths.keys():
                dict_paths[sub] = []

            if (item['phase'] ==  self.cfig['val_phase']):
                partition_IDs['validation'].append(sub)
            if (item['phase'] == self.cfig['tt_phase']):
                partition_IDs['test'].append(sub)

            if (item['phase'] != self.cfig['val_phase'] and item['phase'] != self.cfig['tt_phase']):
                partition_IDs['train'].append(sub)

            dict_paths[sub].append(path)
            if self.cfig['use_dis']:
                dict_dis[sub].append(item['time_dis'])
            if sub not in labels.keys():
                labels[sub] = int(item['gt'])
            else:
                labels[sub] = max(int(item['gt']), labels[sub])
            path_labels[path] = int(item['gt'])

            dict_diags[sub].append(item['diag_dis'])

        if self.cfig['external_test']:
            df = pd.read_csv(self.cfig['ext_csv'])
            list_IDs = list(set(df['subject'].tolist()))  # ID represents subject ID
            list_IDs = [str(i) for i in list_IDs]  # transfer all the ID to string
            ext_ids = []
            #partition_IDs = {'train': [], 'validation': [], 'test': []}
            #labels, dict_paths, = {}, {}  # dict_dists, {}
            #dict_diags, dict_dis = {}, {}
            #path_labels = {}
            for i in range(len(list_IDs)):
                dict_paths[list_IDs[i]] = []
                # dict_dists[list_IDs[i]] = [] # this is the old version
                dict_dis[list_IDs[i]] = []
                dict_diags[list_IDs[i]] = []
            for i, item in df.iterrows():
                path = self.cfig['data_path']['mcl'] + '/' + item['item']
                if not os.path.exists(path):
                    print(path, 'not exist')
                    continue
                sub = str(item['subject'])
                if sub not in dict_paths.keys():
                    dict_paths[sub] = []

                ext_ids.append(sub)

                # if (item['phase'] == self.cfig['val_phase']):
                #     partition_IDs['validation'].append(sub)
                # if (item['phase'] == self.cfig['tt_phase']):
                #     partition_IDs['test'].append(sub)
                #
                # if (item['phase'] != self.cfig['val_phase'] and item['phase'] != self.cfig['tt_phase']):
                #     partition_IDs['train'].append(sub)

                dict_paths[sub].append(path)
                if self.cfig['use_dis']:
                    dict_dis[sub].append(item['time_dis'])
                if sub not in labels.keys():
                    labels[sub] = int(item['gt'])
                else:
                    labels[sub] = max(int(item['gt']), labels[sub])
                path_labels[path] = int(item['gt'])

                dict_diags[sub].append(item['diag_dis'])

        partition_IDs['train'] = list(set(partition_IDs['train']))
        partition_IDs['validation'] = list(set(partition_IDs['validation']))
        partition_IDs['test'] = list(set(partition_IDs['test']))
        ext_ids = list(set(ext_ids))

        assert (len(set(partition_IDs['test']) & set(partition_IDs['validation'])) == 0)
        assert (len(set(partition_IDs['train']) & set(partition_IDs['validation'])) == 0)
        print(set(partition_IDs['test']) & set(partition_IDs['train']))
        assert (len(set(partition_IDs['test']) & set(partition_IDs['train'])) == 0)
        
        if self.cfig['add_positive']:
            print ('========================add more positive =======')
            u=[]                               
            for i in partition_IDs["train"]:
                 if(labels[i]==1): u.append(i)
            partition_IDs['train']+=u

        paramstrain = {'shuffle': True,
                  'num_workers': 4,
                  'batch_size': self.cfig['batch_size']}
        paramstest = {'shuffle': False,
                  'num_workers': 4,
                  'batch_size': self.cfig['test_batch_size']}
        sample_size = self.cfig['sample_size']
        max_step = self.cfig['max_step']
        num_patch = self.cfig['num_patch']
        # if self.cfig['with_global']:
        #     num_patch = self.cfig[self.cfig['model_name']]['in_channel'] - 2
        # else:
        #     num_patch = self.cfig[self.cfig['model_name']]['in_channel']
        print ('sample size is: ', sample_size)
        print ('the number of patches: ', num_patch)
        print ('max step is: ', max_step)
        print ('The total training subjects is ', len(partition_IDs['train']))
        print ('The positive number is ', sum([labels[i] for i in partition_IDs['train']]))
        
        if self.cfig['RnnPatch_loader'] == 'RnnPatch_loader':
            training_set=RnnPatch_loader(partition_IDs['train'], dict_paths, labels, self.cfig['pbb_path'], max_step, sample_size, num_patch, argu = True, with_global = self.cfig['with_global'])
            validation_set=RnnPatch_loader(partition_IDs['validation'], dict_paths, labels, self.cfig['pbb_path'], max_step, sample_size, num_patch, argu = False, with_global = self.cfig['with_global'])
            #test_set = RnnPatch_loader(partition_IDs['test'], dict_paths, labels, self.cfig['pbb_path'], max_step, sample_size, num_patch, argu = False, with_global = self.cfig['with_global'], zeropad_patch = self.cfig['zeropad_patch'])
            
        if self.cfig['RnnPatch_loader'] == 'RnnPatch_loader2':
            training_set=RnnPatch_loader2(partition_IDs['train'], dict_paths, labels,  max_step, sample_size, argu = True)
            validation_set=RnnPatch_loader2(partition_IDs['validation'], dict_paths, labels,  max_step, sample_size, argu = False)
            #test_set = RnnPatch_loader2(partition_IDs['test'], dict_paths, labels,  max_step, sample_size, phase='test')
            
        if self.cfig['RnnPatch_loader'] == 'DisRnnPatch_loader':
            training_set=DisRnnPatch_loader(partition_IDs['train'], dict_paths, dict_dists, labels, self.cfig['pbb_path'], max_step, sample_size, num_patch, argu = True, with_global = self.cfig['with_global'])
            validation_set=DisRnnPatch_loader(partition_IDs['validation'], dict_paths,dict_dists, labels, self.cfig['pbb_path'], max_step, sample_size, num_patch, argu = False, with_global = self.cfig['with_global'])
            #test_set = DisRnnPatch_loader(partition_IDs['test'], dict_paths, dict_dists,labels, self.cfig['pbb_path'], max_step, sample_size, num_patch, argu = False, with_global = self.cfig['with_global'])

        if self.cfig['RnnPatch_loader'] == "RNNMultiS_loader":
            print ("----------------use RNNMultiS_loader ------------------")
            training_set = RNNMultiS_loader(partition_IDs['train'], dict_paths, path_labels, max_step, sample_size, num_patch, argu = True)
            validation_set = RNNMultiS_loader(partition_IDs['validation'], dict_paths, path_labels, max_step, sample_size, num_patch, argu = False)
            test_set = RNNMultiS_loader(partition_IDs['test'], dict_paths, path_labels, max_step, sample_size, num_patch, argu = False)

        if self.cfig['RnnPatch_loader'] == "RNNDiagMultiS_REG_loader":
            print ("----------------use RNNDiagMultiS_REG_loader ------------------")
            training_set = RNNDiagMultiS_REG_loader(partition_IDs['train'], dict_paths, dict_diags, path_labels, max_step, sample_size, num_patch, argu = True)
            validation_set = RNNDiagMultiS_REG_loader(partition_IDs['validation'], dict_paths,dict_diags, path_labels, max_step, sample_size, num_patch, argu = False)
            test_set = RNNDiagMultiS_REG_loader(partition_IDs['test'], dict_paths, dict_diags,path_labels, max_step, sample_size,num_patch, argu=False)

        if self.cfig['RnnPatch_loader'] == "RNNDiagMultiS_loader":
            print ("------- use RNNDiagMultiS_loader -----------")
            training_set = RNNDiagMultiS_loader(partition_IDs['train'], dict_paths, dict_diags, path_labels, max_step, sample_size,
                                            num_patch, argu=True)
            validation_set = RNNDiagMultiS_loader(partition_IDs['validation'], dict_paths, dict_diags, path_labels, max_step,
                                              sample_size, num_patch, argu=False)
            test_set = RNNDiagMultiS_loader(partition_IDs['test'], dict_paths, dict_diags, path_labels, max_step, sample_size,
                                        num_patch, argu=False)
            if self.cfig['external_test']:
                ext_set = RNNDiagMultiS_loader(ext_ids, dict_paths, dict_diags, path_labels, max_step, sample_size, num_patch, argu=False)

        if self.cfig['RnnPatch_loader'] == "RNNDisMultiS_loader":
            print ("------- use RNNDiagMultiS_loader in RNN dis version-----------")
            training_set = RNNDiagMultiS_loader(partition_IDs['train'], dict_paths, dict_dis, path_labels, max_step, sample_size,
                                            num_patch, argu=True)
            validation_set = RNNDiagMultiS_loader(partition_IDs['validation'], dict_paths, dict_dis, path_labels, max_step,
                                              sample_size, num_patch, argu=False)
            test_set = RNNDiagMultiS_loader(partition_IDs['test'], dict_paths, dict_dis, path_labels, max_step, sample_size,
                                        num_patch, argu=False)


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
            if self.cfig['iseval'] and epoch % 3 == 1:
                if self.cfig['external_test']:
                    self.eval_epoch(epoch, 'ext')
                self.eval_epoch(epoch, 'eval')
                self.eval_epoch(epoch, 'test')

                #self.test_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[],[]
        for batch_idx, tup in enumerate(self.train_loader):

            if self.cfig['loss_name'] in ['CURE', 'rank_loss']:
                data, diag_vec, target, paths = tup
                data, diag_vec, target = data.to(self.device), diag_vec.to(self.device), target.to(self.device)
                diag_t0 = diag_vec[:, 0]
                diag_t1 = diag_vec[:, 1]
            elif self.cfig['use_dis']:
                data, dis_vec, target, paths = tup
                data, dis_vec, target = data.to(self.device), dis_vec.to(self.device), target.to(self.device)

            else:
                data, target, paths = tup
                data, target = data.to(self.device), target.to(self.device)
            print (data.shape)
            self.optim.zero_grad()
            data = data.permute([1, 0, 2, 3, 4])
            #print (sum(data))
            # nii_img1 = nib.Nifti1Image(data[0][0].data.cpu().numpy(), affine = np.eye(4))
            # print ('nii_img shape', nii_img1.shape)
            # nib.save(nii_img1, '/nfs/masi/gaor2/tmp/train_nifti/train/' + paths[0] + '0.nii.gz')
            # nii_img2 = nib.Nifti1Image(data[1][0].data.cpu().numpy(), affine=np.eye(4))
            # print('nii_img shape', nii_img2.shape)
            # nib.save(nii_img2, '/nfs/masi/gaor2/tmp/train_nifti/train/'+ paths[0] + '1.nii.gz')
            if batch_idx == 0:
                print (data.shape, 'data shape')

                #print ('point example: ', dist[0])
            # if self.cfig['use_dis']:
            #     feat, pred = self.model(data, dist)
            # else:
            if self.cfig['use_dis']:
                scan_t0, scan_t1, feat, pred = self.model(data, dis_vec)
            else:
                scan_t0, scan_t1, feat, pred = self.model(data)           # here should be careful
            #loss = self.criterion(pred, target)
            if self.cfig['loss_name'] == 'CURE':
                regloss0 = LossPool(pred, target, self.cfig, scan_t= scan_t0, diag_t= diag_t0, loss_name = 'reg_bl').get_loss()
                regloss1 = LossPool(pred, target, self.cfig, scan_t= scan_t1, diag_t= diag_t1, loss_name = 'reg_bl').get_loss()
                celoss = LossPool(pred, target.float(), self.cfig, loss_name= 'bi_ce_loss').get_loss()
                loss = self.cfig['alpha'] * (regloss0 + regloss1) + celoss
            elif self.cfig['loss_name'] == "rank_loss":
                rankloss = LossPool(pred, target, self.cfig, scan_t=scan_t0, diag_t=scan_t1,
                                    loss_name='rank_loss').get_loss()
                regloss = LossPool(pred, target, self.cfig, scan_t=scan_t1, diag_t=diag_t1,
                                   loss_name='reg_bl').get_loss()
                celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()
                loss = self.cfig['alpha'] * (regloss + rankloss) + celoss

            else:
                loss = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            loss.backward()
            self.optim.step()
            print_str = 'train epoch=%d, batch_idx=%d/%d\n' % (
            epoch, batch_idx, len(self.train_loader))
            print(print_str)
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
            csv_info = ['epoch', 'auc','loss', 'accuracy', 'f1', 'recall', 'precision']
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
        
        #---------------------- save to tensorboard ----------------#
#         if self.cfig['save_tensorlog']:
#             self.logger.scalar_summary('loss', np.mean(loss_list), epoch + 1)
#             self.logger.scalar_summary('f1', f1, epoch + 1)
#             self.logger.scalar_summary('accuracy', accuracy, epoch + 1)
#             self.logger.scalar_summary('recall', recall, epoch + 1)
#             self.logger.scalar_summary('precision', precision, epoch + 1)

    def eval_epoch(self, epoch, phase):
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, phase + '.csv')
        pred_list, target_list, loss_list, pos_list = [], [],[],[]
        if phase == 'eval':
            loader = self.val_loader
        if phase == 'test':
            loader = self.test_loader
        if phase == 'ext':
            loader = self.ext_loader
        with torch.no_grad():
            for batch_idx, tup in enumerate(loader):
                if self.cfig['loss_name'] in ['CURE', 'rank_loss']:
                    data, diag_vec, target, paths = tup
                    data, diag_vec, target = data.to(self.device), diag_vec.to(self.device), target.to(self.device)
                    diag_t0 = diag_vec[:, 0]
                    diag_t1 = diag_vec[:, 1]
                elif self.cfig['use_dis']:
                    data, dis_vec, target, paths = tup
                    data, dis_vec, target = data.to(self.device), dis_vec.to(self.device), target.to(self.device)
                else:
                    data, target, paths = tup
                    data, target = data.to(self.device), target.to(self.device)
                data = data.permute([1, 0, 2, 3, 4])
                self.optim.zero_grad()
                # if self.cfig['use_dis']:
                #     feat, pred = self.model(data, dist)
                # else:
                if self.cfig['use_dis']:
                    scan_t0, scan_t1, feat, pred = self.model(data, dis_vec)
                else:
                    scan_t0, scan_t1, feat, pred = self.model(data)

                if self.cfig['loss_name'] == 'CURE':
                    #regloss0 = LossPool(pred, target, self.cfig, scan_t=scan_t0, diag_t=diag_t0,
                    #                    loss_name='reg_bl').get_loss()
                    regloss1 = LossPool(pred, target, self.cfig, scan_t=scan_t1, diag_t=diag_t1,
                                        loss_name='reg_bl').get_loss()
                    celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()
                    loss = self.cfig['alpha'] * regloss1 + celoss
                elif self.cfig['loss_name'] == "rank_loss":
                    rankloss = LossPool(pred, target, self.cfig, scan_t=scan_t0, diag_t = scan_t1, loss_name='rank_loss').get_loss()
                    regloss = LossPool(pred, target, self.cfig, scan_t=scan_t1, diag_t=diag_t1,
                                       loss_name='reg_bl').get_loss()
                    celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()
                    loss = self.cfig['alpha'] * ( rankloss + regloss ) + celoss

                else:
                    loss = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()

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
        
#         #---------------------- save to tensorboard ----------------#
#         if self.cfig['save_tensorlog']:
#             self.logger.scalar_summary('val_loss', np.mean(loss_list), epoch + 1)
#             self.logger.scalar_summary('val_f1', f1, epoch + 1)
#             self.logger.scalar_summary('val_accuracy', accuracy, epoch + 1)
#             self.logger.scalar_summary('val_recall', recall, epoch + 1)
#             self.logger.scalar_summary('val_precision', precision, epoch + 1)
        
    def test_epoch(self, epoch, phase = 'val'):
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
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
        ID_list, t0_list, d0_list,t1_list, d1_list = [], [], [], [], []
        if phase == 'val':
            loader = self.val_loader
        if phase == 'train':
            loader = self.train_loader
        if phase == 'test':
            loader = self.test_loader
        with torch.no_grad():
            for batch_idx, tup in enumerate(loader):
                print(batch_idx)
                if self.cfig['loss_name'] in ['CURE', 'rank_loss']:
                    data, diag_vec, target, paths = tup
                    data, diag_vec, target = data.to(self.device), diag_vec.to(self.device), target.to(self.device)
                    diag_t0 = diag_vec[:, 0]
                    diag_t1 = diag_vec[:, 1]
                else:
                    data, target, paths = tup
                    data, target = data.to(self.device), target.to(self.device)

                print(data.shape)
                self.optim.zero_grad()
                data = data.permute([1, 0, 2, 3, 4])
                scan_t0, scan_t1, feat, pred = self.model(data)

                if self.cfig['loss_name'] == 'CURE':

                    regloss0 = LossPool(pred, target, self.cfig, scan_t=scan_t0, diag_t=diag_t0,
                                        loss_name='reg_bl').get_loss()

                    regloss1 = LossPool(pred, target, self.cfig, scan_t=scan_t1, diag_t=diag_t1,
                                        loss_name='reg_bl').get_loss()

                    celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()

                    loss = self.cfig['alpha'] * (regloss0 + regloss1) + celoss

                elif self.cfig['loss_name'] == "rank_loss":

                    rankloss = LossPool(pred, target, self.cfig, scan_t=scan_t0, diag_t=scan_t1,

                                        loss_name='rank_loss').get_loss()

                    regloss = LossPool(pred, target, self.cfig, scan_t=scan_t1, diag_t=diag_t1,

                                       loss_name='reg_bl').get_loss()

                    celoss = LossPool(pred, target.float(), self.cfig, loss_name='bi_ce_loss').get_loss()

                    loss = self.cfig['alpha'] * (regloss + rankloss) + celoss


                else:

                    loss = LossPool(pred, target.float(), self.cfig, loss_name=self.cfig['loss_name']).get_loss()

                ID_list += paths
                t0_list += scan_t0.data.cpu().numpy().tolist()
                t1_list += scan_t1.data.cpu().numpy().tolist()
                d0_list += diag_t0.data.cpu().numpy().tolist()
                d1_list += diag_t1.data.cpu().numpy().tolist()

                pos_list += pred.data.cpu().numpy().tolist()
                pred_list += (pred > 0.5).tolist()
                target_list += target.data.cpu().numpy().tolist()
                loss_list.append(loss.data.cpu().numpy().tolist())


        print(confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)
        f1 = f1_score(target_list, pred_list)
        recall = recall_score(target_list, pred_list)
        precision = precision_score(target_list, pred_list)
        accuracy = accuracy_score(target_list, pred_list)

        return ID_list, target_list, pos_list, d0_list, t0_list, d1_list, t1_list, roc_auc, f1, recall, precision, accuracy
    
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
        
