import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from func.tools.plotfig import plot_confusion_matrix, plot_roc, plot_testval
from func.models.model import model_define
from func.data.data_generator import *
from func.data.rnn_dataloader import *
import torch.nn.functional as F
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import os

def test_loader( test_label_csv, test_data_root, test_indexs, sample_size, batch_size, mode = 'CNN'):
        df=pd.read_csv(test_label_csv)   
        list_IDs = list(set(df['subject'].tolist()))
        list_IDs = [str(i) for i in list_IDs]
        #partition_IDs={'train':[],'validation':[], 'test':[]}
        test_IDs = []
        labels, dict_paths = {}, {}
        for i in range(len(list_IDs)):
            dict_paths[list_IDs[i]] = []
        for i, item in df.iterrows():
            path= test_data_root + '/' +item['item']
            sub = str(item['subject'])
            if sub not in dict_paths.keys():
                dict_paths[sub] = []
            if (item['trainvalnew'] in test_indexs):
                test_IDs.append(sub)

            dict_paths[sub].append(path)
            labels[sub] = int(item['y'])
        
        test_IDs = list(set(test_IDs))

        paramstest = {'shuffle': False,
                  'num_workers': 4,
                  'batch_size': 2}
        
        print ('sample size is: ', sample_size)
        print ('The total test subjects is ', len(test_IDs))
        print ('The positive number is ', sum([labels[i] for i in test_IDs]))
        if mode == 'CNN':
            test_set = CNNDataset(test_IDs, dict_paths, labels, sample_size, argu = False)
        if mode == "RNN":
            max_step = 3
            test_set = RNNDataset(test_IDs, dict_paths, labels, max_step, sample_size, argu = False)
            
        if mode == 'Patch':
            pbb_path = ['/share5/gaor2/data/MCL/DSB_File/bbox_regcombine']
            test_set = Patch_loader(test_IDs, dict_paths, labels, pbb_path, sample_size, num_patch = 5, argu = False)
        if mode == 'RnnPatch':
            max_step =3
            pbb_path = ['/share5/gaor2/data/MCL/DSB_File/bbox_regcombine']
            test_set = RnnPatch_loader(test_IDs, dict_paths, labels, pbb_path, max_step, sample_size, argu = False)
            
        test_generator = data.DataLoader(test_set, **paramstest)
        return test_generator

    
    
def test_epoch(model, model_pth, test_loader, save_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        #if device == 'cuda': #there is a GPU device
        model.load_state_dict(torch.load(model_pth))
        #else:
        #    model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
        pred_list, target_list, loss_list, sub_list = [],[],[], []
        feature0_list, feature1_list, pos_list = [], [], []
        for batch_idx, (data, target, sub_name) in enumerate(test_loader):  # no test_loader yet
            
           # data = data.permute([1, 0, 2, 3, 4,5])           # this one is for RNN
            
            data, target = data.to(device), target.to(device)
           # self.optim.zero_grad()
            pred = model(data)             # here should be careful
            pred_prob = F.softmax(pred)
            #loss = self.criterion(pred, target)
#            loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            pred_cls = pred.data.max(1)[1]     # not test yet
            pred_list+=pred_cls.data.cpu().numpy().tolist()
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            target_list+=target.data.cpu().numpy().tolist()
            for i in range(len(sub_name)):
                #sub = item.split('time')[0]
                sub_list.append(sub_name[i])
                #target_list.append(target[i].data.cpu().numpy())
                feature = pred_prob[i].data.cpu().numpy().tolist()
                feature0_list.append(feature[0])
                feature1_list.append(feature[1])
#            loss_list.append(loss.data.cpu().numpy().tolist())
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)  
        f1=f1_score(target_list,pred_list)
        recall=recall_score(target_list,pred_list)
        precision=precision_score(target_list,pred_list)
        accuracy=accuracy_score(target_list,pred_list)
        data = pd.DataFrame()
        data['subject'] = sub_list
        data['feature0'] = feature0_list
        data['feature1'] = feature1_list
        data['target'] = target_list
#        data['trainvalnew'] = df['trainvalnew']
        data.to_csv(save_path)
        return roc_auc, accuracy, f1, recall, precision

class Tester(object):                                                             
    def __init__(self,  cfig, testval = 'val', method_name = None):
        self.cfig = cfig
        self.testval = testval
        self.method_name = method_name
        
    def strings_to_method(self):
        method = getattr(self, self.method_name)
        return method()
    def kaggle(self):                           
        print ('hello')
        df = pd.read_csv(self.cfig['kaggle_path']) # here may should change
        if self.testval == 'test':
            df = df[df["trainvalnew"] == 0]
        if self.testval == 'val':
            df = df[df["trainvalnew"] == 4]
        ground = []
        pred = []
        for i, item in df.iterrows():
            if (item["from_baseline"] == item["from_baseline"]):
                ground.append(item["y"])
                pred.append(item["from_baseline"])
        predprob = np.asarray(pred)
        fpr, tpr, threshold = metrics.roc_curve(ground, pred)
        optimal_idx = np.argmax(tpr - fpr)
        thresholdval = threshold[optimal_idx]
        pred = predprob > thresholdval
        if (self.testval == 'test'):
            pred = predprob > 0.44079244
        ground = np.asarray(ground)
        plot_roc(ground, predprob)
        plot_testval(ground, pred)
        
    
    def clinical_only(self):
        df = pd.read_csv(self.cfig['clinical_path'])

        train = df.query("trainvalnew == 1 or trainvalnew ==2 or trainvalnew == 3")
        val = df.loc[df['trainvalnew'] == 4]
        test = df.loc[df['trainvalnew'] == 0]
        need_item = ['Gender', 'BMI', 'History', 'Age Started Cig', 'Age Quit', 'Cigs Per Day', 'Pack Years']
        label_item = 'y'
        train_Y = train[label_item]
        train_X = train[need_item]

        val_Y = val[label_item]
        val_X = val[need_item]

        test_Y = test[label_item]
        test_X = test[need_item]

        
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_val = xgb.DMatrix(val_X, label=val_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        watchlist = [(xg_train, 'train'), (xg_val, 'val')]
        bst = xgb.train(self.cfig, xg_train, self.cfig['n'], watchlist, early_stopping_rounds=self.cfig['esr'])
        pred_prob = bst.predict(xg_val, ntree_limit=bst.best_ntree_limit)
        fpr, tpr, threshold = metrics.roc_curve(val_Y, pred_prob)

        optimal_idx = np.argmax(tpr - fpr)
        thresholdval = threshold[optimal_idx]
        pred = pred_prob > thresholdval

        error_rate = np.sum(pred != val_Y) / val_Y.shape[0]
        print('VAL error using softprob = {}'.format(error_rate))


        if ('val' == self.testval):
            print ('plot roc-----')
            plot_roc(val_Y, pred_prob)
            plot_testval(val_Y, pred)
        pred_prob = bst.predict(xg_test, ntree_limit=bst.best_ntree_limit)
        fpr, tpr, threshold = metrics.roc_curve(test_Y, pred_prob)

        thresholdtest = thresholdval
        pred = pred_prob > thresholdtest

        error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
        print('TEST error using softprob = {}'.format(error_rate))
        if ('test' == self.testval):
            print ('plot roc-----')
            plot_roc(test_Y, pred_prob)
            plot_testval(test_Y, pred)
        xgb.plot_importance(bst)


    def deepfeat_only(self):
        data = pd.read_csv(self.cfig['feature_path']) 
        if self.testval == 'val':
            df = data.loc[data['trainvalnew'] == 4]
        if self.testval == 'test':
            df = data.loc[data['trainvalnew'] == 0]
        target = df['target']
        pred_prob, pred = [], []
        
        for i,item in df.iterrows():
            pred_prob.append(np.exp(float(item['feature1'])))
            pred.append(1 if float(item['feature1']) > float(item['feature0']) else 0)
        
        plot_testval(target, pred, save_path=None)
        np.save('/share5/gaor2/tmp_file/target2.npy', target)
        np.save('/share5/gaor2/tmp_file/pred_prob2.npy', pred_prob)
        plot_roc(target, pred_prob)

    def deep_and_clinical(self):
        if not os.path.exists(self.cfig['combine_path']):
            clicl = pd.read_csv(self.cfig['clinical_path'])
            feat = pd.read_csv(self.cfig['feature_path'])
            data = pd.merge(clicl, feat, on = ['subject', 'trainvalnew'], how = 'inner')
            data.to_csv(self.cfig['combine_path'])
        else:
            data = pd.read_csv(self.cfig['combine_path'])
        
        train = data.query("trainvalnew == 1 or trainvalnew ==2 or trainvalnew == 3")
        val = data.loc[data['trainvalnew'] == 4]
        test = data.loc[data['trainvalnew'] == 0]
        need_item = ['Gender', 'BMI', 'History', 'Age Started Cig', 'Age Quit', 'Cigs Per Day', 'Pack Years', 'feature0', 'feature1']
        label_item = 'y'
        train_Y = train[label_item]
        train_X = train[need_item]
        
        val_Y = val[label_item]
        val_X = val[need_item]

        test_Y = test[label_item]
        test_X = test[need_item]

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_val = xgb.DMatrix(val_X, label=val_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        watchlist = [(xg_train, 'train'), (xg_val, 'val')]

        # logistic classifier
        bst = xgb.train(self.cfig, xg_train, self.cfig['n'], watchlist, early_stopping_rounds=self.cfig['esr'])
        pred_prob = bst.predict(xg_val, ntree_limit=bst.best_ntree_limit)
        fpr, tpr, threshold = metrics.roc_curve(val_Y, pred_prob)

        optimal_idx = np.argmax(tpr - fpr)
        thresholdval = threshold[optimal_idx]
        pred = pred_prob > thresholdval

        error_rate = np.sum(pred != val_Y) / val_Y.shape[0]
        print('VAL error using softprob = {}'.format(error_rate))
        xgb.plot_importance(bst)

        if ('val' in self.testval):
            plot_testval(val_Y, pred)

            plot_roc(val_Y, pred_prob)

        pred_prob = bst.predict(xg_test, ntree_limit=bst.best_ntree_limit)
        fpr, tpr, threshold = metrics.roc_curve(test_Y, pred_prob)

        thresholdtest = thresholdval
        pred = pred_prob > thresholdtest

        error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
        print('TEST error using softprob = {}'.format(error_rate))

        if ('test' in self.testval):
            plot_testval(test_Y, pred)

            plot_roc(test_Y, pred_prob)







