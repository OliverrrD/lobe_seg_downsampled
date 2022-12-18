import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from func.tools.tool import weighted_center
from func.loss.angle_margin import *

class BFocalLoss(nn.Module):
    #https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(BFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class ReFocalLoss(nn.Module):
    #https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    def __init__(self, alpha=2, gamma=2, logits=False, reduce=True):
        super(ReFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha /( 1 + 10 * (1-pt)**self.gamma) * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def weighted_binary_cross_entropy(output, target, weights=None):
        #print (target)
        #print (output)

        if weights is not None:
            assert len(weights) == 2

            loss = weights[1] * (target.cuda().float() * torch.log(output)) + \
                   weights[0] * ((1. - target.cuda().float()) * torch.log(1. - output))
        else:
            loss = target.cuda() * torch.log(output) + (1 - target.cuda()) * torch.log(1. - output)

        return torch.neg(torch.mean(loss))

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets):
        
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print('class_mask',class_mask)  # like one-hot 
        #print ('self.alpha ', self.alpha)
        #print (inputs.is_cuda, self.alpha.is_cuda)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]
        #print (ids.data.view(-1))
        #print (alpha, self.alpha)

        probs = (P * class_mask).sum(1).view(-1, 1)
        #print ('probs', probs)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        #print (alpha.shape, log_p.shape)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p # self.gamma.float().to(self.device)
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
class RegSmoothCE(nn.Module): 
    def __init__(self, class_num, alpha = None, alpha_reg = 0.01, size_average = True):
        super(RegSmoothCE, self).__init__()
        if alpha is None:
            print ('-----1-------')
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                print ('----2-----')
                self.alpha = alpha
            else:
                print ('------3------')
                self.alpha = alpha * Variable(torch.ones(class_num, 1))
        self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, inputs, targets, scan_t, diag_t):
        print ('scan_t', scan_t)
        print ('diag_t', diag_t)
        it0 = nn.MSELoss()(scan_t, diag_t)
        it2 = nn.CrossEntropyLoss()(inputs, targets)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print('class_mask',class_mask)  # like one-hot 
        #print ('self.alpha ', self.alpha)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        alpha_vec = 2 * alpha.float() * ( targets.unsqueeze(1).float() - 0.5)
        #print (alpha_vec)
        #print ("P: ", P)

        probs = (P * class_mask).sum(1).view(-1, 1)
        #print ('probs', probs)

        log_p = probs.log()
        #print (alpha_vec.shape, alpha.shape, targets.unsqueeze(1).shape, log_p.shape)
        it1 = - alpha_vec * (diag_t)  * log_p
        #print (it1)
        #it1 = -alpha * (torch.pow((1 - probs), self.gamma.float())) * log_p # 
        
        if self.size_average:
            it1_loss = it1.mean()
        else:
            it1_loss = it1.sum()
        print (it0, it1_loss, it2)
        batch_loss = it0 + it1_loss + it2
        return batch_loss


class DisBCE(nn.Module):
    def __init__(self, param_a, param_c, mode='const'):
        super(DisBCE, self).__init__()
        self.a = param_a
        self.c = param_c
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def forward(self, inputs, targets, diag_t):
        assert inputs.shape == targets.shape
        assert targets.shape == diag_t.shape
        if self.mode == 'const':
            coff = self.a * torch.exp(- self.c * diag_t)
        if self.mode == 'log':
            coff = self.a
        if self.mode == 'linear':
            coff = self.a * diag_t * torch.exp(- self.c * diag_t * diag_t)
        coff = coff.float()
        input_copy = inputs.clone()
        ones_device = torch.ones(len(diag_t)).to(self.device)
        non_cancer_index = (targets <= 0.5).nonzero()
        #print(coff)
        #print(targets)
        coff[non_cancer_index] = ones_device[non_cancer_index]
        input_copy[non_cancer_index] = 1 - inputs[non_cancer_index]
        input_copy = torch.max(input_copy, 0.001 * ones_device)
        log_prob = torch.log(input_copy)

        #print(log_prob)
        loss = torch.mean(- coff * log_prob)
        return loss

class DisBCE1(nn.Module):
    def __init__(self, param_a, param_c, mode='const'):
        super(DisBCE1, self).__init__()
        self.a = param_a
        self.c = param_c
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def forward(self, inputs, targets, diag_t):

        print(inputs)
        assert inputs.shape == targets.shape
        assert targets.shape == diag_t.shape
        ones_device = torch.ones(len(diag_t)).to(self.device).double()
        if self.mode == 'const':
            gt = torch.exp(- self.c * diag_t)
        if self.mode == 'sqrt':
            gt = torch.max(ones_device, torch.exp(- self.c * torch.sqrt(diag_t)))
        if self.mode == 'linear':
            gt = diag_t * torch.exp(- self.c * diag_t * diag_t)
        threshold = (1 + gt) * 0.5
        ident = ones_device.clone()
        ident[inputs > threshold] = 0
        input_copy = inputs.clone()

       # input_copy = ident.float() * input_copy

        non_cancer_index = (targets <= 0.6).nonzero()
        ident[non_cancer_index] = 1
        print ("ident", ident)

        print("target ", targets)

        input_copy[non_cancer_index] = 1 - inputs[non_cancer_index]
        print (input_copy)
        log_prob = ident.float() * torch.log(input_copy)

        loss = torch.mean(- log_prob)
        return loss

class RankLoss(nn.Module):            # haven't been tested 0102
    def __init__(self, rank_mode = 'exp'):
        super(RankLoss, self).__init__()
        self.mode = rank_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, t0, t1):
        assert t0.shape == t1.shape
        try:
            assert t1.shape == targets.shape
        except:
            print (t1, targets)
        assert t1.shape == targets.shape
        zero_device = torch.zeros(len(t0)).to(self.device)
        if self.mode == 'exp':
            diff = torch.max(zero_device, torch.exp(- 0.1 * t0) - torch.exp(-0.1 * t1))
            loss = nn.L1Loss(reduce = True, size_average=True)(diff, zero_device)
        if self.mode == 'l2':
            #print ('------- ranking use l2 ----------')
            diff = torch.max(zero_device, t1 - t0)
            loss = nn.MSELoss(reduce=True, size_average=True)(diff, zero_device)
        return loss


class RegRisk(nn.Module):
    def __init__(self, lamb=None, margin=1, size_average=True, reg_mode='l2'):
        super(RegRisk, self).__init__()
        self.lamb = lamb
        self.margin = margin
        # self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = reg_mode

    def forward(self, inputs, targets, scan_t, diag_t):

        assert scan_t.shape == diag_t.shape
        assert scan_t.shape == targets.shape
        ones_device = self.margin * torch.ones(len(scan_t)).to(self.device)
        zero_device = torch.zeros(len(scan_t)).to(self.device)


        diff = (scan_t - diag_t)
        diff_neg = (scan_t - diag_t - ones_device)

        diff_neg_copy = self.lamb * diff_neg.clone()
        non_cancer = torch.max(zero_device, diff_neg_copy)  # here should be different from BLReg

     #   cancer_neg_t = torch.max(zero_device, diff.clone())

        non_cancer_index = (targets <= 0.5).nonzero()
        diff[non_cancer_index] = non_cancer[non_cancer_index]

  #      cancer_index = ((targets > 0.5)).nonzero()

        #diff[cancer_index] = diff[cancer_index]

        if self.mode == 'l2':
            it1 = nn.MSELoss(reduce=True, size_average=True)(diff, zero_device)
            # it1 = torch.mean(diff * diff)
        if self.mode in ['l1', 'exp']:
            print("------------------------use l1 or exp loss ------------------")
            it1 = nn.L1Loss(reduce=True, size_average=True)(diff, zero_device)
        if self.mode == 'mix':  # 0112
            it1 = nn.L1Loss(reduce=True, size_average=True)(torch.exp(0.1 * diff),
                                                            torch.ones(len(scan_t)).to(self.device))

        return it1


class RegBL(nn.Module):
    def __init__(self, lamb = None, margin = 1, size_average = True, reg_mode = 'l2', bicensor = False):
        super(RegBL, self).__init__()
        self.lamb = lamb
        self.margin = margin
        #self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = reg_mode
        self.bicensor = bicensor

    def forward(self, inputs, targets, scan_t, diag_t):
#         print ('scan_t, diag_t, targets', scan_t.shape, diag_t.shape)
# #         print ('diag_t', )
#         print ('targets', targets.shape)
        assert scan_t.shape == diag_t.shape
        assert scan_t.shape == targets.shape
        #it0 = nn.CrossEntropyLoss()(inputs, targets)
        ones_device = self.margin * torch.ones(len(scan_t)).to(self.device)
        #lamb_device = self.lamb * torch.ones(len(scan_t)).to(self.device)
        zero_device = torch.zeros(len(scan_t)).to(self.device)

 #       focal = abs(scan_t.detach() - diag_t)  # add 20200110

        if self.mode == 'exp':
            print("---------------- use  exp ----------------------------")
            diff =  torch.exp(-0.1 * diag_t + 0.1 * ones_device) - torch.exp(- 0.1 * scan_t) #focal *
            diff_neg = torch.exp(- 0.1 * diag_t - 0.1 * ones_device) - torch.exp(- 0.1 * scan_t)
        else:
            #print ("-- use focal ------")
            diff =  (scan_t - diag_t + ones_device) #focal *
            diff_neg = (scan_t - diag_t - ones_device)

        diff_neg_copy =  self.lamb * diff_neg.clone()
        non_cancer = torch.min(zero_device, diff_neg_copy)
        #cancer_neg_t = torch.max(zero_device, scan_t + ones_device)
        cancer_neg_t = torch.max(zero_device, diff.clone())  # this should also work
        
        non_cancer_index = (targets <= 0.5).nonzero()
        diff[non_cancer_index] = non_cancer[non_cancer_index]
        
        if self.bicensor:
            cancer_neg_t_index = ((targets > 0.5)).nonzero()
        else:
            cancer_neg_t_index = ((targets > 0.5) & (diag_t < self.margin)).nonzero()  # diag_t < self.margin 20200102
          # all censored


        diff[cancer_neg_t_index] = cancer_neg_t[cancer_neg_t_index]
        #print ('diff: ', diff)
        if self.mode == 'l2':
            it1 = nn.MSELoss(reduce=True, size_average=True)(diff, zero_device)
            #it1 = torch.mean(diff * diff)
        if self.mode in ['l1', 'exp']:
            print ("------------------------use l1 or exp loss ------------------")
            it1 = nn.L1Loss(reduce = True, size_average=True)(diff, zero_device)
        if self.mode == 'mix':  # 0112
            it1 = nn.L1Loss(reduce = True, size_average = True)(torch.exp(0.1*diff), torch.ones(len(scan_t)).to(self.device))

        #print ('reg loss:', it1.data.cpu().numpy())

        return it1


    
class RegBL0715(nn.Module): #
    def __init__(self, alpha = None, margin = 1, size_average = True):
        super(RegBL, self).__init__()
        self.alpha = alpha
        self.margin = margin
        #self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        
    def forward(self, inputs, targets, scan_t, diag_t):
#         print ('scan_t, diag_t, targets', scan_t.shape, diag_t.shape)
#         print ('diag_t', )
#         print ('targets', targets.shape)
        assert scan_t.shape == diag_t.shape
        assert scan_t.shape == targets.shape
        it0 = nn.CrossEntropyLoss()(inputs, targets)
        ones_device = self.margin * torch.ones(len(scan_t)).to(self.device)
        zero_device = torch.zeros(len(scan_t)).to(self.device)
        diff = scan_t - diag_t + ones_device
        diff_neg = scan_t - diag_t - ones_device
        diff_neg_copy = diff_neg.clone()
        
        non_cancer = torch.min(zero_device, diff_neg_copy)
        cancer_neg_t = torch.max(zero_device, scan_t + ones_device)
        
        non_cancer_index = (targets <= 0.5).nonzero()
        diff[non_cancer_index] = non_cancer[non_cancer_index]
        
        cancer_neg_t_index = ((targets > 0.5) & (diag_t < -self.margin)).nonzero()
        diff[cancer_neg_t_index] =  cancer_neg_t[cancer_neg_t_index]
        it1 = torch.mean(diff * diff)
        
        batch_loss = it0 + self.alpha * it1
        print ('cross entropy: ',it0.data.cpu().numpy(), 'reg loss:', it1.data.cpu().numpy(), 'total loss:', batch_loss.data.cpu().numpy())
        return it1, batch_loss
    
class RegBL20190711(nn.Module):
    def __init__(self, alpha = None, pos_weight = 1, size_average = True):
        super(RegBL, self).__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        #self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        
    def forward(self, inputs, targets, scan_t, diag_t):
#         print ('scan_t, diag_t, targets', scan_t.shape, diag_t.shape)
#         print ('diag_t', )
#         print ('targets', targets.shape)
        assert scan_t.shape == diag_t.shape
        assert scan_t.shape == targets.shape
        it0 = nn.CrossEntropyLoss()(inputs, targets)
        diff = scan_t - diag_t
        diff_copy = diff.clone()
        ones_device = torch.ones(len(scan_t)).to(self.device)
        non_cancer = torch.min(ones_device, diff_copy)
        cancer_neg_t = torch.max(- ones_device, scan_t + ones_device)
        
        non_cancer_index = (targets <= 0.5).nonzero()
        diff[non_cancer_index] = non_cancer[non_cancer_index]
        
        cancer_neg_t_index = ((targets > 0.5) & (diag_t < -1)).nonzero()
        diff[cancer_neg_t_index] =  cancer_neg_t[cancer_neg_t_index]
        it1 = torch.mean(diff * diff)
        
        batch_loss = it0 + self.alpha * it1
        print ('cross entropy: ',it0.data.cpu().numpy(), 'reg loss:', it1.data.cpu().numpy(), 'total loss:', batch_loss.data.cpu().numpy())
        return it1, batch_loss

# class AggregateCE(nn.Module):
#     """the aggregated Cross Entropy"""
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#     def forward(self, *input: Any, **kwargs: Any):
    
        
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        torch.manual_seed(666)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc()
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        #print ('---------batch_size_tensor---------', batch_size_tensor)
        #print (feat, label, self.centers)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss

class CenterlossFunc(nn.Module):
    def __init__(self):
        super(CenterlossFunc, self).__init__()
    def forward(self, feature, label, centers, batch_size):
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

class RGCenterLoss(nn.Module):
#     exp(||x_i - c_i||) - log(||x_i - c_j||) + CE
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(RGCenterLoss, self).__init__()
        torch.manual_seed(666)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = RGCenter()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        
        loss = self.centerlossfunc(feat, label, self.centers, batch_size, self.num_classes)
        return loss    
    
class RGCenter(nn.Module):
    def __init__(self):
        super(RGCenter, self).__init__()
    def forward(self, feature, label, centers, batch_size, num_classes):
        repeat_centers = centers.unsqueeze(0).repeat(batch_size, 1, 1)
        repeat_feature = feature.unsqueeze(1).repeat(1, num_classes, 1)
        assert repeat_centers.shape == repeat_feature.shape
        all_log = torch.log((repeat_feature - repeat_centers).pow(2).sum(-1) + 1).sum()
        centers_batch = centers.index_select(0, label.long())
        center_log = torch.log((feature - centers_batch).pow(2).sum(-1) + 1).sum()
        #center_exp = torch.exp((feature - centers_batch).pow(2).sum(-1) + 1).sum()
        center_ = (feature - centers_batch).pow(2).sum()
        #print ('all_log', all_log)
        #print ('center_log', center_log)
        #print ('center_exp', center_exp)
        #print ('centers_batch', centers_batch)
        #print ('feature - centers_batch', (feature - centers_batch).pow(2).sum(-1))
        return center_ / 2.0 / batch_size - (all_log - center_log) / 2.0 / (num_classes - 1) / batch_size 
    
class LabelSmoothing(nn.Module):
    def __init__(self, cfid, size_average = True):
        super(LabelSmoothing, self).__init__()
        self.cfid = cfid
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print (class_mask)
        class_mask = abs( 1 - self.cfid - class_mask)
        #print ('class_mask', class_mask)
        log_p = P.log()
        log_p = (log_p * class_mask).sum(1).view(-1, 1)
        
        if self.size_average:
            loss = -log_p.mean()
        else:
            loss = -log_p.sum()
        return loss
    
# def MutuInfo(nn.Module):
#     '''
#     create one parameter layer to adapt the dependence.  a * pred_color + b = color
#    '''
    
# class CenterLoss(nn.Module):
 # ## a easy way define Center Loss: https://discuss.pytorch.org/t/trainable-variable-in-loss-function-and-matrix-multiplication/495/4 
#     def __init__(self,center_num,feature_dim):
#         super(CenterLoss,self).__init__()
#         self.center_num = center_num
#         self.feature_dim = feature_dim
#         self.center_features = nn.Parameter(torch.Tensor(self.center_num,self.feature_dim))
#         nn.init.normal(self.center_features.data,mean=0,std=0.1)

#     def forward(self,x,label):
#         B = x.size()[0]
#         center = torch.index_select(self.center_features,0,label)
#         diff = x-center
#         loss = diff.pow(2).sum() / B
#         return loss    