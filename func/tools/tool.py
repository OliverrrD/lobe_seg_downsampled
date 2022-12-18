import sys
sys.path.append('..')

import torch
from torch.utils import data
from torch.autograd import Variable
from scipy.spatial import distance
import torch.nn.functional as F
from func.models.Net_3D_conv1 import net_conv1
from func.models.layers import HookBasedFeatureExtractor
import scipy
import os
import numpy as np
import pandas as pd



def extractFeat0(model_path, csv_path, img_root, out_dim, with_softmax = True):
    '''
    huge_frame = extractFeat("/share5/wangj46/DeepLungNet/tmp20/cv4/models/model_epoch_0060.pth",
            "/share5/wangj46/fromlocal/demographic/final_4fold.csv",
            "/share5/wangj46/fromlocal/318box/img/", 2, with_softmax = True)
    :param model_path:
    :param csv_path:
    :param img_root:
    :param out_dim:
    :param with_softmax:
    :return:
    '''
    df = pd.read_csv(csv_path)
    partition = {'validation': []}
    labels = {}
    for i, item in df.iterrows():
        path = img_root + '/' + item['share3_namepath']
        partition['validation'].append(path)
        labels[path] = item['y']
    params = {'shuffle': False,
              'num_workers': 8,
              'batch_size': 1}
    validation_set = Dataset(partition['validation'], labels, phase='test')
    validation_generator = data.DataLoader(validation_set, **params)
    model = net_conv1(
		nonlocal_mode='concatenation_softmax',
		aggregation_mode='concat',
		in_channel = 2,
		n_classes = 2,
	    )
    model.load_state_dict(torch.load(model_path))
    #out_dim = 2

    model.eval()
    model = model.cuda()
    torch.cuda.manual_seed(1)

    huge_frame = pd.DataFrame()
    for i in range(out_dim):
        huge_frame[i] = None
    for batch_idx, (d, target, sub_name) in enumerate(validation_generator):
        d, target = d.cuda(), target.cuda()
        d, target = Variable(d, volatile=True), Variable(target, volatile=True)
        if with_softmax:
            with torch.no_grad():
                pred_prob = model(d)
            pred_clss = F.log_softmax(pred_prob)  ###
            #pred = pred_clss.data.max(1)[1]  # get th
        else:
            with torch.no_grad():
                feature_extractor = HookBasedFeatureExtractor(model, "classifier", False)
                fmap=feature_extractor.forward(d)
                feature = fmap[0][0].data.cpu().numpy() # of shape (4,256)


        for i in range(pred_clss.shape[0]):  # pred_clss or pred_prob??
            huge_frame.loc[sub_name[i][sub_name[i].rfind("/") + 1:]] = pred_clss[i].data.cpu().numpy().reshape(
                [out_dim, ]) 
    return huge_frame

def extractFeat(model_path, csv_path, save_path, with_softmax = True):
    device = 'cuda'
    df = pd.read_csv(csv_path)
    sub_list = []
    #path_list = df['path'].tolist()
    labels = {}
    for i, item in df.iterrows():
        path = item['path'][:-7]
        sub_list.append(path)
        labels[path] = item['y']
    params = {'shuffle': False,
                  'num_workers': 4,
                  'batch_size': 1}
    data_set = Dataset(sub_list, labels, phase='test')
    data_generator = torch.utils.data.DataLoader(data_set, **params)
    model = net_conv1(
            nonlocal_mode='concatenation_softmax',
            aggregation_mode='concat',
            in_channel=2,
            n_classes=2,
        )
    if device == 'cuda': #there is a GPU device
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))

    model.eval()
    model = model.to(device)
    torch.cuda.manual_seed(1)
    sub_list = []
    feature0_list = []
    feature1_list = []
    target_list = []
    for batch_idx, (d, target, sub_name) in enumerate(data_generator):
        d, target = d.to(device), target.to(device)
        pred_prob = model(d)
        pred_clss = F.log_softmax(pred_prob)
        assert len(pred_clss) == len(sub_name)
        for i in range(len(sub_name)):
            item = sub_name[i].split('/')[-1]
            sub = item.split('time')[0]
            #print (sub, pred_clss[i].data.cpu().numpy().tolist())
            sub_list.append(sub)
            target_list.append(target[i].data.cpu().numpy())
            feature = pred_clss[i].data.cpu().numpy().tolist()
            feature0_list.append(feature[0])
            feature1_list.append(feature[1])
    data = pd.DataFrame()
    data['subject'] = sub_list
    data['feature0'] = feature0_list
    data['feature1'] = feature1_list
    data['target'] = target_list
    data['trainvalnew'] = df['trainvalnew']
    data.to_csv(save_path)



def saveOneImg(img,path,cate_name,sub_name,surfix,):
    filename = "%s-x-%s-x-%s.png"%(cate_name,sub_name,surfix)
    file = os.path.join(path,filename)
    scipy.misc.imsave(file, img)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def get_distance(target,score,ind,Threshold=0.75):
    dist_list = []
    coord_list = []
    target_coord_list = []
    weight_coord_list = []
    for i in range(target.size()[1]):
        targetImg = target[ind,i,:,:].data.cpu().numpy()
        scoreImg = score[ind,i,:,:].data.cpu().numpy()
        targetCoord = np.unravel_index(targetImg.argmax(),targetImg.shape)
        scoreCoord = np.unravel_index(scoreImg.argmax(),scoreImg.shape)
        # grid = np.meshgrid(range(score.size()[2]), range(score.size()[3]), indexing='ij')
        # x0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        # y0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        #
        y0,x0 = weighted_center(score[ind,i,:,:],Threshold)

        weightCoord = (x0.data.cpu().numpy()[0],y0.data.cpu().numpy()[0])
        distVal = distance.euclidean(scoreCoord,targetCoord)
        dist_list.append(distVal)
        coord_list.append(scoreCoord)
        target_coord_list.append(targetCoord)
        weight_coord_list.append(weightCoord)
    return dist_list,coord_list,target_coord_list,weight_coord_list

def save_images(results_epoch_dir,data,sub_name,cate_name,pred_lmk,target=None):
    saveOneImg(data[0, 0, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_trueGray")
    for i in range(pred_lmk.size()[1]):
        saveOneImg(pred_lmk[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_pred%d" % (i))
        if not (target is None):
            saveOneImg(target[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_true%d" % (i))

def weighted_center(input,threshold=0.75):
    # m= torch.nn.Tanh()
    # input = m(input)

    input = torch.add(input, -input.min().expand(input.size())) / torch.add(input.max().expand(input.size()), -input.min().expand(input.size()))
    m = torch.nn.Threshold(threshold, 0)
    input = m(input)
    # if input.sum()==0:
    #     input=input
    # mask_ind = input.le(0.5)
    # input.masked_fill_(mask_ind, 0.0)
    grid = np.meshgrid(range(input.size()[0]), range(input.size()[1]), indexing='ij')
    x0 = torch.mul(input, Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / input.sum()
    y0 = torch.mul(input, Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / input.sum()
    return x0, y0

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)