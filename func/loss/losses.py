
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from func.tools.tool import weighted_center
from func.loss.angle_margin import *
from func.loss.functions import *
import pdb

class LossPool(object):
    '''
    https://pytorch.org/docs/0.3.0/nn.html#loss-functions
    please add all these loss to here.
    self.pred.shape = (N x D)
    self.target.shape = (N,)
    '''
    def __init__(self, pred, target, cfig, feat = None, scan_t = None, diag_t = None, loss_name = None):
        self.loss_name = loss_name
        self.pred = pred
        self.target = target
        self.cfig = cfig
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scan_t = scan_t
        self.diag_t = diag_t
        self.feat = feat
    def get_loss(self):
        method = getattr(self, self.loss_name)
        return method()
    
    def nll_loss(self):
        print ('===========using   nll_loss================')
        return nn.NLLLoss()(self.pred, self.target)

    def smthl1_loss(self):
        #print ('==========using smooth l1 loss ==============')
        return torch.nn.SmoothL1Loss()(self.pred, self.target)

    def cross_entropy_loss(self):
        return nn.CrossEntropyLoss()(self.pred, self.target)
    
    def bi_ce_loss(self):
        return nn.BCELoss()(self.pred, self.target)

    def weight_bceloss(self):
        loss = weighted_binary_cross_entropy(self.pred, self.target, self.cfig['bce_weight'])
        return loss
    
    def focal_loss(self):
        print('===========using  focal_loss================')
        #alpha = torch.from_numpy(np.array(self.cfig['focal_alpha'], dtype = np.float32))
        #gamma = torch.from_numpy(np.array(self.cfig['focal_gamma'], dtype = np.float32))
        #print (alpha)
        focal = FocalLoss(class_num = 2)
        loss = focal(self.pred, self.target)
        return loss

    def bi_focal_loss(self):
        focal = BFocalLoss(alpha=1, gamma=2, logits=False, reduce=True)
        loss = focal(self.pred, self.target)
        return loss
    
    def re_focal_loss(self):
        focal = ReFocalLoss(alpha=2, gamma=2, logits=False, reduce=True)
        loss = focal(self.pred, self.target)
        return loss
    
    def reg_smoothing(self):
        print ('--------- use reg smoothing loss ---------')
        #alpha = torch.from_numpy(np.array(self.cfig['alpha']))
        reg= RegSmoothCE(self.pred.shape[1], self.cfig['lamb'], self.cfig['alpha_reg'])
        loss = reg(self.pred, self.target, self.scan_t, self.diag_t)
        return loss
    
    def label_smoothing(self):
        #print ('--------- use label smoothing loss ---------')
        funct= LabelSmoothing(self.cfig['cfid'])
        loss = funct(self.pred, self.target)
        return loss
    
    def center_loss(self):
        print ('-------add center loss -----------')
        funct = CenterLoss(self.cfig['num_class'], self.cfig['feat_dim'])
        
        print ('num of class: ', self.cfig['num_class'])
        print ('dim of feature: ', self.cfig['feat_dim'])
        loss = funct(self.target, self.feat) # the self.pred should be feature here.
        print (self.target, self.feat)
        return loss
    
    def reg_bl(self):
        funct = RegBL(lamb = self.cfig['lamb'], margin = self.cfig['margin'], reg_mode = self.cfig['reg_mode'], bicensor = self.cfig['bicensor'])
        loss = funct(self.pred, self.target, self.scan_t, self.diag_t)
        return loss

    def reg_risk(self):
        funct = RegRisk(lamb = self.cfig['risk_lamb'], margin = self.cfig['risk_margin'], reg_mode = self.cfig['risk_mode'])
        loss = funct(self.pred, self.target, self.scan_t, self.diag_t)
        return loss
    
    def disbce_loss(self):
        print ('------ use distanced BCE loss -------')
        funct = DisBCE(param_a = self.cfig['param_a'], param_c = self.cfig['param_c'])
        loss = funct(self.pred, self.target, self.diag_t)
        return loss

# ======================================================================#
# -------I think the below functions all need further test ------------ #  
# ======================================================================#
    def rank_loss(self):
        funct = RankLoss(rank_mode = self.cfig['rank_mode'])
        loss = funct(self.pred, self.target, self.scan_t, self.diag_t)
        return loss

    def disbce_loss1(self):
        print ("-----------use distanced bce loss 1 --------")
        funct = DisBCE1(param_a = self.cfig['param_a'], param_c = self.cfig['param_c'])
        loss = funct(self.pred, self.target, self.diag_t)
        return loss
    

    def RGcenter_loss(self):
        print ('-------add RG center loss ---------')
        funct = RGCenterLoss(self.cfig['num_class'], self.cfig['feat_dim'])
        print ('num of class: ', self.cfig['num_class'])
        print ('dim of feature: ', self.cfig['feat_dim'])
        loss = funct(self.target, self.feat) # the self.pred should be feature here.
        print (self.target, self.feat)
        return loss
   
    def RPEE(self):       #haven't test yet
        """
        from paper: Regularizing Prediction Entropy Enhances Deep Learning with Limited Data
        :return:
        """
        return nn.CrossEntropyLoss()(self.pred, self.target) - self.cfig['RPEE_alpha'] * nn.CrossEntropyLoss(self.pred, self.pred)

    def arc_loss(self):
        #print('===========using   arc_loss================')
        metric_fc = ArcMarginProduct(self.cfig['feat_dim'], self.cfig['n_classes'], s=30, m=0.5, easy_margin= False).to(self.device)
        pdict = metric_fc(self.pred, self.target)
        loss = nn.CrossEntropyLoss()(pdict, self.target)
        return loss, pdict
    
    def addm_loss(self):
        metric_fc = AddMarginProduct(512, self.cfig['n_classes'], s=30, m=0.35).to(self.device)
        pdict = metric_fc(self.pred, self.target)
        loss = nn.CrossEntropyLoss()(pdict, self.target)
        return loss
    
    def sphere_loss(self):
        metric_fc = SphereProduct(512, self.cfig['n_classes'], m=4).to(self.device)
        pdict = metric_fc(self.pred, self.target)
        loss = nn.CrossEntropyLoss()(pdict, self.target)
        return loss
    
    def label_smoothing_old(self):            #haven't test successfully, the difference of KLDivLoss and CrossEntropyLoss not clear. But can learn from focal_loss. 
#         num_class = self.pred.shape[1]
#         target = self.target.data.clone()
#         y = torch.eye(num_class) 
#         target = y[target]      # convert to one hot
#         smooth = 1 - self.cfig['conf']
#         target.fill_(smooth / (num_class - 1))
#         target.scatter_(1, self.target.data.unsqueeze(1), self.cfig['conf'])
#         print ('label smoothing target: ', target)
#         return nn.KLDivLoss()(self.pred, target)
        it1 =  nn.CrossEntropyLoss()(self.pred, self.target)
        it2 =  (1 - self.cfig['cfid']) * torch.mean(F.log_softmax(self.pred)) # used to be 1. it's wrong for this item.
        return it1 + it2
    
    def reg_smoothing_CE_old(self): # d_t: diagnose time, s_t: scan time
        it1 = (self.cfig['alpha1'] * (self.cfig['p_t'] - self.cfig['g_t']) * (self.cfig['p_t'] - self.cfig['g_t']) + 1) * nn.CrossEntropyLoss()(self.pred, self.target)
        it2 = (1 - self.cfig['alpha2'] * (self.cfig['p_t'] - self.cfig['p_t']) -  self.cfig['alpha2']) * torch.mean(F.log_softmax(self.pred, 1)) 
        return it1 + it2

class Test_LossPool(object):
    def __init__(self, pred, target, cfig, scan_t = None, diag_t = None, loss_name = None):
        self.pred = pred
        self.target = target
        self.loss_name = loss_name
        self.cfig = cfig
        self.scan_t = scan_t
        self.diag_t = diag_t
    def get_loss(self):
        method = getattr(self, self.loss_name)
        return method()

    def nll_loss(self):
        num_sample = self.pred.shape[0]

        sum = 0
        for i in range(num_sample):
            sum -= self.pred[i][self.target[i]]
        return sum / num_sample
    
    def reg_bl(self):
        it2 = nn.CrossEntropyLoss()(self.pred, self.target)
        num_class = self.pred.shape[1]
        num_sample = self.pred.shape[0]
        it1 = 0
        for i in range(num_sample):
            assert self.target[i] == 0 or self.target[i] == 1
            if self.target[i] == 0:
                it1 += torch.pow(min(torch.Tensor([0]).float(), self.scan_t[i] - self.diag_t[i]), 2)
            if self.target[i] == 1:
                if self.diag_t[i] > 0:
                    it1 += torch.pow(self.scan_t[i] - self.diag_t[i], 2)
                if self.diag_t[i] <= 0:
                    #pdb.set_trace()
                    it1 += torch.pow(max(torch.Tensor([0]).float(), self.scan_t[i]),2)
        it1 = it1 / num_sample
        print ('the cross entropy is', it2, ' and the bl reg loss is', it1)
        return self.cfig['alpha'] * it1 + it2
        
    def disbce_loss(self):
        loss = 0
        num_sample = self.pred.shape[0]
        print ('the number of samples: ', num_sample)
        for i in range(num_sample):
            if self.target[i] == 1:
                loss -= self.cfig['param_a'] * torch.exp( - self.cfig['param_c'] * self.diag_t[i]) * torch.log(self.pred[i])
            else:
                assert self.target[i] == 0
                loss -= torch.log(1 - self.pred[i])
        return loss / num_sample

    def disbce_new_loss(self):
        loss = 0
        num_sample = self.pred.shape[0]
        print ('the number of samples: ', num_sample)
        for i in range(num_sample):
            if self.target[i] == 1:
                loss -= self.cfig['param_a'] * torch.log(self.pred[i])  # please continue here
            else:
                assert self.target[i] == 0
                loss -= torch.log(1 - self.pred[i])
        return loss / num_sample
    
    def cross_entropy_loss(self):
        # equation comes from http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
        first = 0
        num_class = self.pred.shape[1]
        num_sample = self.pred.shape[0]
        print ('the number of samples: ', num_sample, "the number of class", num_class)
        first, second = [0] * num_sample, [0] * num_sample
        for i in range(num_sample):
            first[i] -= self.pred[i][int(self.target[i])]

        for i in range(num_sample):
            for j in range(num_class):
                second[i] += math.exp(self.pred[i][j])
        res = 0
        for i in range(num_sample):
            res += first[i] + math.log(second[i])
        return res / num_sample
    
    def reg_smoothing(self):
        print ('------- test reg smoothing ------')
        num_class = self.pred.shape[1]
        num_sample = self.pred.shape[0]
        print ('the number of samples: ', num_sample, "the number of class", num_class)
        prob = F.softmax(self.pred)
        sum = 0
        for i in range(num_sample):
            p = prob[i][int(self.target[i])]
            if self.target[i] == 1:
                it1_m = self.cfig['alpha']
            else:
                it1_m = - self.cfig['alpha']
            sum -= it1_m * (self.scan_t[i] - self.diag_t[i]) * (self.scan_t[i] - self.diag_t[i]) * p.log()     
        it1 = 1.0 * sum[0] / num_sample
        it0 = nn.MSELoss()(self.scan_t, self.diag_t)
        it2 = nn.CrossEntropyLoss()(self.pred, self.target)
        return (it0 + it1 + it2)

    def focal_loss(self):
        print('===========using   focal_loss================')
        num_class = self.pred.shape[1]
        num_sample = self.pred.shape[0]
        print ('the number of samples: ', num_sample, "the number of class", num_class)
        prob = F.softmax(self.pred)
        sum = 0
        for i in range(num_sample):
            p = prob[i][int(self.target[i])]
            sum -= self.cfig['focal_alpha'] * torch.pow((1 - p), self.cfig['focal_gamma']) * p.log()
        return sum / num_sample
    
    def label_smoothing(self):
        target = torch.zeros(len(self.target), dtype = torch.float)
        for i in range(len(self.target)):
            if self.target[i] == 0:
                target[i] = 1 - self.cfig['cfid']
                
            else:
                target[i] = self.cfig['cfid']
                
        s = 0
        self.pred = F.softmax(self.pred)
        print (self.pred)
        self.pred = self.pred.log()
        print (self.pred)
        
        print (target, self.cfig['cfid'])
        for i in range(len(self.pred)):
            s +=  (- target[i] * self.pred[i][1] - (1 - target[i]) * self.pred[i][0])
        
        return s / len(self.pred)
    
    def center_loss(self):
        num_sample = len(self.target)
        num_class = self.cfig['num_class']
        feat_dim = self.cfig['feat_dim']
        torch.manual_seed(666)
        centers = torch.randn((num_class, feat_dim))
        centers_batch = centers.index_select(0, self.target.long())
        return (self.pred - centers_batch).pow(2).sum() / 2.0 / num_sample
    
    def RGcenter_loss(self):
        num_sample = len(self.target)
        num_class = self.cfig['num_class']
        feat_dim = self.cfig['feat_dim']
        torch.manual_seed(666)
        centers = torch.randn((num_class, feat_dim))
        centers_batch = centers.index_select(0, self.target.long())






