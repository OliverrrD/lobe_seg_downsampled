import torch
from torch.utils import data
import numpy as np
import nibabel as nib
import os
import random
import skimage as sk
from skimage import transform, util
from func.data.data_tool import nms
from func.data.image_manipulation import *
import pdb

class GetLoader(data.Dataset):
    def __init__(self, loader_name, list_IDs, **cfig):
        assert loader_name in ['cnndataset', 'cnndiag', 'twod_loader', 'twohalf_loader', 'patch_loader', 'singlescan', 'multis_loader', 'multis_diag', 'multis_diag_reg']
        self.loader_name = loader_name
        self.cfig = cfig
        self.cfig['list_IDs'] = list_IDs
    def get_loader(self):
        method = getattr(self, self.loader_name)
        return method()
        
    def cnndataset(self):
        return CNNDataset(**self.cfig)
    def cnndiag(self):
        return CNNDiag(**self.cfig)
    def twod_loader(self):
        return TwoD_loader(**self.cfig)
    def twohalf_loader(self):
        return Twohalf_loader(**self.cfig)
    def patch_loader(self):
        return Patch_loader(**self.cfig)
    def singlescan(self):
        return SingleScan(**self.cfig)
    def multis_loader(self):
        return MultiS_loader(**self.cfig)
    def multis_diag(self):
        return MultiS_Diag(**self.cfig)
    def multis_diag_reg(self):
        return MultiS_Diag_REG(**self.cfig)
    
class SingleScan(data.Dataset):
    def __init__(self, list_IDs, data_path):
        self.data_path = data_path
        self.data = nib.load(self.data_path).get_data()   
    def __len__(self):
        shape = self.data.shape
        return (shape[1] // 2 ) * (shape[2] //2 )
    
    def __getitem__(self, index): 
#        x = self.data
        sample_size = 224
        half_n = 24
        img = self.data
        shape = self.data.shape
        box = [0, 0, 0]
        box[0] = 122
        box[1] = int(2 * (index / (shape[1] // 2 )))
        box[2] = int(2 * (index % (shape[1] // 2 ))) 
        if box[1] >= shape[1]: box[1] = shape[1] - 1
        if box[2] >= shape[2]: box[2] = shape[2] - 1
        
        tmp_img_a = img[box[0]]
        #print (box)
        tmp_mask_a = np.zeros(tmp_img_a.shape)
        
        tmp_mask_a[max(0,box[1] - half_n): box[1] + half_n, max(0,box[2] - half_n): box[2] + half_n] = 1
        tmp_img_c = img[:, box[1], :]
        tmp_mask_c = np.zeros(tmp_img_c.shape)
        tmp_mask_c[max(0,box[0] - half_n): box[0] + half_n, max(0,box[2] - half_n): box[2] + half_n] = 1
        tmp_img_s = img[:, :, box[2]]
        tmp_mask_s = np.zeros(tmp_img_s.shape)
        tmp_mask_s[max(0,box[0] - half_n): box[0] + half_n, max(0,box[1] - half_n): box[1] + half_n] = 1
        img_a = transform.resize(tmp_img_a, [sample_size, sample_size], mode='edge', preserve_range='True')
        mask_a = transform.resize(tmp_mask_a, [sample_size, sample_size], mode='edge', preserve_range='True')
        img_c = transform.resize(tmp_img_c, [sample_size, sample_size], mode='edge', preserve_range='True')
        mask_c = transform.resize(tmp_mask_c, [sample_size, sample_size], mode='edge', preserve_range='True')
        img_s = transform.resize(tmp_img_s, [sample_size, sample_size], mode='edge', preserve_range='True')
        mask_s = transform.resize(tmp_mask_s, [sample_size, sample_size], mode='edge', preserve_range='True')
        x = np.zeros((6, 224, 224))
        x[0] = img_a
        x[1] = mask_a
        x[2] = img_c
        x[3] = mask_c
        x[4] = img_s
        x[5] = mask_s
        x = x.astype('float32')
        return x, (box[0], box[1], box[2])

class thorax_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_labels, sample_size, argu = False, lastscan = True, use_mask = False):
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size
        self.argu = argu
        self.lastscan = lastscan
        self.all_paths = self.get_all_IDs()
        self.use_mask = use_mask

    def get_all_IDs(self):
        all_paths = []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)
        return all_paths

    def __len__(self):
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)

    def __getitem__(self, index):

        if self.lastscan == True:
            ID = self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            y = self.path_labels[path]
        else:
            path = self.all_paths[index]
            y = self.path_labels[path]

        img_file = path
        img_3d = nib.load(img_file)
        img = img_3d.get_data()
        if self.use_mask:
            mask_file = img_file.replace("img", "mask")
            mask_3d = nib.load(mask_file)
            mask = mask_3d.get_data()
  
        if self.argu:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-15, 15)
                seed = np.random.randint(4, size=6)
                choice = np.random.choice(2, 3)
                
                if self.use_mask:
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)
                else:
                    img = transform.rotate(img, random_degree, preserve_range=True)
                    img, img_ignore = random_translation_pair(img, img, seed, choice)

        if self.use_mask: 
            x = np.zeros((2, self.sample_size, self.sample_size, self.sample_size))
            x[0] = img
            
        else:
            x = np.zeros((1, self.sample_size, self.sample_size, self.sample_size))
            x[0] = img
            x[1] = mask

        x = x.astype('float32')

        return x, y, path.split('/')[-1]
    
class CNNDataset(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_labels, sample_size, argu = False, use_mask = True, lastscan = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size
        self.use_mask = use_mask
        self.argu = argu
        self.lastscan = lastscan
        self.all_paths = self.get_all_IDs()

    def get_all_IDs(self):
        all_paths =  []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)
                
        return all_paths
        
    def __len__(self):
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)
    
    def __getitem__(self, index):
        
        if self.lastscan == True:
            ID =self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            y = self.path_labels[path]
        else:
            path = self.all_paths[index]
            y = self.path_labels[path]

        img_file = path
        mask_file = path.replace("img","mask") 
        
        img_3d = nib.load(img_file)
        img0 = img_3d.get_data()
        #print (img0.shape)

        #new_shape = [self.sample_size, int(0.8 * self.sample_size), self.sample_size]
        new_shape = [self.sample_size,  self.sample_size, self.sample_size]
        img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
        img = np.transpose(img, (2, 0, 1))
        
        if self.use_mask:
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
            mask = np.transpose(mask, (2, 0, 1))
        
        img = (img - img.min()) / (img.max() - img.min())

        if self.argu:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-15, 15)
                seed = np.random.randint(4, size=6)
                choice = np.random.choice(2, 3)
                if self.use_mask:
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)

        img = img * 255.0
        if self.use_mask:
            mask = mask * 255.0
        
        if self.use_mask: 
            #x = np.zeros((1, self.sample_size, self.sample_size,  self.sample_size))
            mask = mask.astype('float32')
           # x[0] = img * (0.5 + mask * 0.002 )
            x = np.zeros((2, self.sample_size, self.sample_size,  self.sample_size))

            x[0] = img
            x[1] = mask
        else:
            x = np.zeros((1, self.sample_size, self.sample_size, self.sample_size))

            x[0] = img

        x = x.astype('float32')

        return x, y, path.split('/')[-1]
    
class CNNDiag(data.Dataset):
    def __init__(self, list_IDs, dict_paths, dict_diags, labels, path_labels, sample_size, argu = False, use_mask = True, lastscan = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.dict_diags = dict_diags
        self.sample_size = sample_size
        self.use_mask = use_mask
        self.lastscan = lastscan
        self.path_labels = path_labels
        self.argu = argu
        if not lastscan:
            self.all_paths, self.all_diags = self.get_all_IDs()
        
    def get_all_IDs(self):
        all_paths, all_diags = [], []
        for ID in self.list_IDs:
            tmp_paths = sorted(self.dict_paths[ID])
            tmp_diags = sorted(self.dict_diags[ID], reverse = True)
            for i in range(len(tmp_paths)):
                all_paths.append(tmp_paths[i])
                #all_labels.append(self.labels[ID])
                all_diags.append(tmp_diags[i])
        return all_paths, all_diags #, all_labels
        
    def __len__(self):
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)
    
    def __getitem__(self, index):
        if self.lastscan == True:
            ID =self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            diag = sorted(self.dict_diags[ID], reverse = True)[-1]
            y = self.path_labels[path]
            y_reg = self.labels[ID]
        else:
            path = self.all_paths[index]
            diag = self.all_diags[index]
            y = self.path_labels[path]
            tmp_id = str(int(path.split('/')[-1].split('t')[0]))
            y_reg = self.labels[tmp_id]

        img_file = path
        mask_file = path.replace("img","mask") 
        
        img_3d = nib.load(img_file)
        img0 = img_3d.get_data()

        new_shape = [self.sample_size, self.sample_size, self.sample_size]
        img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
        img = np.transpose(img, (2, 0, 1))
        
        if self.use_mask:
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
            mask = np.transpose(mask, (2, 0, 1))
        
        img = (img - img.min()) / (img.max() - img.min())

        if self.argu:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-15, 15)
                seed = np.random.randint(4, size=6)
                choice = np.random.choice(2, 3)
                if self.use_mask:
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)

        img = img * 255.0
        if self.use_mask:
            mask = mask * 255.0
        
        if self.use_mask: 
            x = np.zeros((2, self.sample_size, self.sample_size, self.sample_size))

            x[0, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img[0:self.sample_size, 0:self.sample_size,
                                                                      0:self.sample_size]
            x[1, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = mask[0:self.sample_size, 0:self.sample_size,
                                                                      0:self.sample_size]
        else:
            x = np.zeros((1, self.sample_size, self.sample_size, self.sample_size))

            x[0, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img[0:self.sample_size, 0:self.sample_size,
                                                                      0:self.sample_size]

        x = x.astype('float32')

        return x, diag, y, y_reg, path.split('/')[-1]

class CNNSurv(data.Dataset):
    def __init__(self, list_IDs, dict_paths, dict_factors, path_labels,  sample_size, argu=False, use_mask=True, lastscan=True):
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size
        self.use_mask = use_mask
        self.argu = argu
        self.lastscan = lastscan
        self.all_paths = self.get_all_IDs()
        self.dict_factors = dict_factors

    def get_all_IDs(self):
        all_paths = []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)
        return all_paths

    def __len__(self):
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        if self.lastscan == True:
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
        else:
            path = self.all_paths[index]
        y = self.path_labels[path]
        factors = np.array(self.dict_factors[path]).astype('float32')

        img_file = path
        mask_file = path.replace("img", "mask")

        img_3d = nib.load(img_file)
        img0 = img_3d.get_data()
        # print (img0.shape)

        # new_shape = [self.sample_size, int(0.8 * self.sample_size), self.sample_size]
        new_shape = [self.sample_size, self.sample_size, self.sample_size]
        img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
        img = np.transpose(img, (2, 0, 1))

        if self.use_mask:
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
            mask = np.transpose(mask, (2, 0, 1))

        img = (img - img.min()) / (img.max() - img.min())

        if self.argu:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-15, 15)
                seed = np.random.randint(4, size=6)
                choice = np.random.choice(2, 3)
                if self.use_mask:
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)

        img = img * 255.0
        if self.use_mask:
            mask = mask * 255.0

        if self.use_mask:
            # x = np.zeros((1, self.sample_size, self.sample_size,  self.sample_size))
            mask = mask.astype('float32')
            # x[0] = img * (0.5 + mask * 0.002 )
            x = np.zeros((2, self.sample_size, self.sample_size, self.sample_size))

            x[0] = img
            x[1] = mask
        else:
            x = np.zeros((1, self.sample_size, self.sample_size, self.sample_size))

            x[0] = img

        x = x.astype('float32')
        #print ('factors', factors.shape)

        return x, factors, y, path.split('/')[-1]

class TwoD_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_labels, sample_size, num_patch,  argu = False, lastscan = False, use_mask = True, ave = False, threed = True, argu_3d = False):
        'Initialization'
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.argu = argu
        self.sample_size = sample_size
        self.dict_paths = dict_paths
        self.num_patch = num_patch
        self.lastscan = lastscan
        self.use_mask = use_mask
        self.all_paths = self.get_all_IDs()
        self.ave = ave
        self.threed = threed
        self.argu_3d = argu_3d
        
    def get_all_IDs(self):
        all_paths =  []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)        
        return all_paths

    def __len__(self):
        'Denotes the total number of samples'
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)
        
    def swap(self, a, index1, index2):
        x = a.copy()
        tmp1 = a[ 2* index1: 2* index1 + 2]
        tmp2 = a[  2*index2: 2* index2 +2]
        x[2 * index1: 2 * index1 + 2] = tmp2
        x[  2* index2: 2* index2  + 2] = tmp1
        return x
    
    def __getitem__(self, index):
        
        if self.lastscan == True:
            ID =self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            y = self.path_labels[path]
            #print ('use lastscan')
        else:
            path = self.all_paths[index]
            y = self.path_labels[path]

        img_file = path
        
#       pbb = np.load(os.path.join(self.pbb_roots, path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        
        try:
            img_3d = nib.load(img_file)
        except:
            print (img_file, ' has problem !')
        img0 = img_3d.get_data()
 #       print ('img0.shape: ', img0.shape)
        if self.use_mask:
            mask_file = path.replace("img","mask") 
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0
        
                              # 5 means we select the most like nodule region
        
        # ===============random parameters: the sequence should be same ==============#

        if self.argu:
            rand = random.uniform(0, 1)
                
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-15, 15)
                seed = np.random.randint(4, size=4)
                choice = np.random.choice(2, 2)
                if self.use_mask:
                    img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                    img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)
        
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
        img0 = img0 * 255.0
        
        channels = self.num_patch
        
#         if self.ave:
            
#             x = np.zeros((channels, self.sample_size, self.sample_size))
        
#             for i in range(channels):         #self.num_patch           #please be careful about the img and img0  

#                 tmp_img = img0[i]   # simulation use i, other may use 3 * i
#                 tmp_mask = mask0[i] 
#                 img = transform.resize(tmp_img, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
#                 mask = transform.resize(tmp_mask, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
#                 mask = mask.astype('float32')
#                 x[i] = img * (0.5 + mask * 0.002)
#         else:

        x = 170 * np.zeros((2 * channels, self.sample_size, self.sample_size))

        for i in range(channels):         #self.num_patch           #please be careful about the img and img0
            if self.threed:
                tmp_img = img0[i] # simulation use i, other may use 3 * i
                tmp_mask = mask0[i]
            else:
                tmp_img = img0[3 * i] # simulation use i, other may use 3 * i
                tmp_mask = mask0[3 * i]
            img = transform.resize(tmp_img, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            mask = transform.resize(tmp_mask, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            x[2*i] = img
            x[2*i + 1] = mask
            
        if self.argu_3d:
            assert x.shape[0] == 18
            index1, index2 = np.random.randint(0,3,2)
            #index1, index2 = 2, 1
            for i in range(3):
                x[i*6: i * 6 + 6] = self.swap(x[i*6: i * 6 + 6], index1, index2)
 
#         index1, index2 = 0,2
#         for i in range(3):
#             #print (x.shape)
#             x[i*6: i * 6 + 6] = self.swap(x[i*6: i * 6 + 6], index1, index2)
#         nii_img = nib.Nifti1Image(x.data, affine = np.eye(4))   
#         nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + path.split('/')[-1] + '.nii.gz')
#         x = z
        x = x.astype('float32')
        return x, y, path.split('/')[-1]

class MultiS_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_labels, sample_size, num_patch, argu = False, lastscan = False, use_mask = True, ave = False, threed = True, argu_3d = False):
        'Initialization'
        self.path_labels = path_labels  
        self.list_IDs = list_IDs
        self.argu = argu
        self.sample_size = sample_size
        self.dict_paths = dict_paths
        self.num_patch = num_patch
        self.lastscan = lastscan
        self.use_mask = use_mask
        self.all_paths = self.get_all_IDs()
        self.ave = ave
        self.threed = threed
        self.argu_3d = argu_3d
        
    def get_all_IDs(self):
        all_paths =  []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)        
        return all_paths

    def __len__(self):
        'Denotes the total number of samples'
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)
    
    def __getitem__(self, index):
        if self.lastscan == True:
            ID =self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            y = self.path_labels[path]
        else:
            path = self.all_paths[index]
            y = self.path_labels[path]

        img_file = path
        
        try:
            img_3d = nib.load(img_file)
        except:
            print (img_file, ' has problem !')
        img0 = img_3d.get_data()
 #       print ('img0.shape: ', img0.shape)
        if self.use_mask:
            mask_file = path.replace("img","mask") 
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0  
        if self.argu:
            rand = random.uniform(0, 1)
                
            if (0 <= rand < 0.6):
                random_degree = random.uniform(-30, 30)
                seed = np.random.randint(4, size=4)
                choice = np.random.choice(2, 2)
                if self.use_mask:
                    img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                    img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)
        
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
        img0 = img0 * 255.0
        
        channels = self.num_patch 
            
        
        #print (img0.shape, channels)
        if self.ave:
            x = 170 * np.zeros((channels, self.sample_size, self.sample_size),dtype = np.uint8)
        else:
            x = 170 * np.zeros((2 * channels, self.sample_size, self.sample_size),dtype = np.uint8)
        #if not self.threed: channels = channels // 15 # nodule * slice
        #print (channels)
        for i in range(channels // 5):   #self.num_patch     #please be careful about the img and img0
            if self.threed:  
                tmp_img = img0[5 * i: 5 * i + 5]   # simulation use i, other may use 3 * i
                tmp_mask = mask0[5 * i: 5 * i + 5]
            else:
                tmp_img = img0[15 * i : 15 * i + 5]  # simulation use i, other may use 3 * i
                tmp_mask = mask0[15 * i : 15 * i + 5]
            #print (self.sample_size, self.sample_size)
            #img = transform.resize(tmp_img, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            #mask = transform.resize(tmp_mask, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            if self.ave:
                
                x[5*i: 5*i + 5] = tmp_img * (0.5 + tmp_mask / 255.0 * 0.5)
                #x[10*i + 5: 10 * i + 10] = tmp_mask
            else:
                x[10*i: 10*i + 5] = tmp_img
                x[10*i + 5: 10 * i + 10] = tmp_mask
            
#         if self.argu_3d:
#             assert x.shape[0] == 18
#             index1, index2 = np.random.randint(0,3,2)
#             #index1, index2 = 2, 1
#             for i in range(3):
# #                 x[i*6: i * 6 + 6] = self.swap(x[i*6: i * 6 + 6], index1, index2)
#         nii_img = nib.Nifti1Image(x.data, affine=np.eye(4))
#         nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + path.split('/')[-1] + '.nii.gz')
#         x = z
        x = x.astype('float32')
        return x, y, path.split('/')[-1]


class MultiS_Diag_REG(data.Dataset):
    def __init__(self, list_IDs, dict_paths, dict_diags, labels, path_labels, sample_size, num_patch, argu=False,
                 lastscan=False, use_mask=True, ave=False, threed=True, argu_3d=False):
        'Initialization'
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.argu = argu
        self.sample_size = sample_size
        self.dict_paths = dict_paths
        self.num_patch = num_patch
        self.lastscan = lastscan
        self.use_mask = use_mask
        self.dict_diags = dict_diags
        self.all_paths = self.get_all_IDs()
        self.ave = ave
        self.threed = threed
        self.argu_3d = argu_3d
        self.labels = labels

    def get_all_IDs(self):
        all_paths, all_diags = [], []
        for ID in self.list_IDs:
            tmp_paths = sorted(self.dict_paths[ID])
            tmp_diags = sorted(self.dict_diags[ID], reverse=True)
            for i in range(len(tmp_paths)):
                all_paths.append(tmp_paths[i])
                # all_labels.append(self.labels[ID])
                all_diags.append(tmp_diags[i])
        return all_paths, all_diags  # , all_labels

    def __len__(self):
        'Denotes the total number of samples'
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)

    def __getitem__(self, index):
        if self.lastscan == True:
            ID = self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            diag = sorted(self.dict_diags[ID], reverse=True)[-1]
            y = self.path_labels[path]
            y_reg = self.labels[ID]
        else:
            path = self.all_paths[index]
            diag = self.all_diags[index]
            y = self.path_labels[path]
            tmp_id = str(int(path.split('/')[-1].split('t')[0]))
            y_reg = self.labels[tmp_id]

        img_file = path

        try:
            img_3d = nib.load(img_file)
        except:
            print(img_file, ' has problem !')
        img0 = img_3d.get_data()
        if self.use_mask:
            mask_file = path.replace("img", "mask")
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0
        if self.argu:
            rand = random.uniform(0, 1)

            if (0 <= rand < 0.8):
                random_degree = random.uniform(-40, 40)
                seed = np.random.randint(10, size=4)
                choice = np.random.choice(2, 2)
                if self.use_mask:
                    img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                    img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)

        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
        img0 = img0 * 255.0

        channels = self.num_patch

        x = 170 * np.zeros((2 * channels, self.sample_size[0], self.sample_size[1]), dtype=np.uint8)

        for i in range(channels // 5):  # self.num_patch     #please be careful about the img and img0
            tmp_img = img0[5 * i: 5 * i + 5]  # simulation use i, other may use 3 * i
            tmp_mask = mask0[5 * i: 5 * i + 5]

            x[10 * i: 10 * i + 5] = tmp_img
            x[10 * i + 5: 10 * i + 10] = tmp_mask

        x = x.astype('float32')
        return x, diag, y, y_reg, path.split('/')[-1]


class MultiS_Diag(data.Dataset):
    def __init__(self, list_IDs, dict_paths, dict_diags, labels, path_labels, sample_size, num_patch, argu = False, lastscan = False, use_mask = True, ave = False, threed = True, argu_3d = False):
        'Initialization'
        self.path_labels = path_labels  
        self.list_IDs = list_IDs
        self.argu = argu
        self.sample_size = sample_size
        self.dict_paths = dict_paths
        self.num_patch = num_patch
        self.lastscan = lastscan
        self.use_mask = use_mask
        self.dict_diags = dict_diags
        self.all_paths = self.get_all_IDs()
        self.ave = ave
        self.threed = threed
        self.argu_3d = argu_3d
        self.labels = labels

        
    def get_all_IDs(self):
        all_paths, all_diags = [], []
        for ID in self.list_IDs:
            tmp_paths = sorted(self.dict_paths[ID])
            tmp_diags = sorted(self.dict_diags[ID], reverse = True)
            for i in range(len(tmp_paths)):
                all_paths.append(tmp_paths[i])
                #all_labels.append(self.labels[ID])
                all_diags.append(tmp_diags[i])
        return all_paths, all_diags #, all_labels

    def __len__(self):
        'Denotes the total number of samples'
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)
    
    def __getitem__(self, index):
        if self.lastscan == True:
            ID =self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one
            diag = sorted(self.dict_diags[ID], reverse = True)[-1]
            y = self.path_labels[path]
            y_reg = self.labels[ID]
        else:
            path = self.all_paths[index]
            diag = self.all_diags[index]
            y = self.path_labels[path]
            tmp_id = str(int(path.split('/')[-1].split('t')[0]))
            y_reg = self.labels[tmp_id]

        img_file = path
        
        try:
            img_3d = nib.load(img_file)
        except:
            print (img_file, ' has problem !')
        img0 = img_3d.get_data()
 #       print ('img0.shape: ', img0.shape)
        if self.use_mask:
            mask_file = path.replace("img","mask") 
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0  
        if self.argu:
            rand = random.uniform(0, 1)
                
            if (0 <= rand < 0.8):
                random_degree = random.uniform(-40, 40)
                seed = np.random.randint(10, size=4)
                choice = np.random.choice(2, 2)
                if self.use_mask:
                    img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                    img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)
        
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
        img0 = img0 * 255.0
        
        channels = self.num_patch 

        x = 170 * np.zeros((2 * channels, self.sample_size, self.sample_size),dtype = np.uint8)
        #print (img0.shape, channels)
        
        #if not self.threed: channels = channels // 15 # nodule * slice
        #print (channels)
        for i in range(channels // 5):   #self.num_patch     #please be careful about the img and img0
            if self.threed:  
                tmp_img = img0[5 * i: 5 * i + 5]   # simulation use i, other may use 3 * i
                tmp_mask = mask0[5 * i: 5 * i + 5]
            else:
                tmp_img = img0[15 * i : 15 * i + 5]  # simulation use i, other may use 3 * i
                tmp_mask = mask0[15 * i : 15 * i + 5]
            #print (self.sample_size, self.sample_size)
            #img = transform.resize(tmp_img, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            #mask = transform.resize(tmp_mask, [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            
            x[10*i: 10*i + 5] = tmp_img
            x[10*i + 5: 10 * i + 10] = tmp_mask
            
#         if self.argu_3d:
#             assert x.shape[0] == 18
#             index1, index2 = np.random.randint(0,3,2)
#             #index1, index2 = 2, 1
#             for i in range(3):
# #                 x[i*6: i * 6 + 6] = self.swap(x[i*6: i * 6 + 6], index1, index2)
#         nii_img = nib.Nifti1Image(x.data, affine=np.eye(4))
#         nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + path.split('/')[-1] + '.nii.gz')
#         x = z
        x = x.astype('float32')
        return x, diag, y, y_reg, path.split('/')[-1]
    
class Twohalf_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_labels, pbb_roots = ['/nfs/masi/NLST/DSB_File/diag/bbox'], sample_size = 160, num_patch = 3, num_slice = 15, argu = False, lastscan = True, argu_3d = False, ave = True):
        'Initialization'
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.argu = argu
        self.sample_size = 160
        self.dict_paths = dict_paths
        self.num_patch = 3
        self.lastscan = True
        self.all_paths = self.get_all_IDs()
        self.num_slice = 15
        self.pbb_roots = pbb_roots

        
    def get_all_IDs(self):
        all_paths =  []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)        
        return all_paths

    def __len__(self):
        'Denotes the total number of samples'
        if self.lastscan:
            return len(self.list_IDs)
        else:
            return len(self.all_paths)
    
    def __getitem__(self, index):
        if self.lastscan == True:
            ID =self.list_IDs[index]
            path = sorted(self.dict_paths[ID])[-1]  # select the lastest one 
            y = self.path_labels[path]
        else:
            path = self.all_paths[index]
            y = self.path_labels[path]
        
#       pbb = np.load(os.path.join(self.pbb_roots, path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        for pbb_root in self.pbb_roots:
            #try:
                pbb = np.load(os.path.join(pbb_root, path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
            #except:
            #    pbb = 

        
        img_3d = nib.load(path)
        img0 = img_3d.get_data()

        
        mask_file = path.replace("img","mask") 
        mask_3d = nib.load(mask_file)
        mask0 = mask_3d.get_data()
        mask0 = mask0 * 255.0
        
        pbb = pbb[pbb[:,0]>-1]
        pbb = nms(pbb,0.05)                                     # used to be 0.05
        #print (len(pbb), ' len of pbb')
        #print (pbb)
        
        boxes = pbb[:self.num_patch]                         # 5 means we select the most like nodule region
        if self.argu:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-20, 20)
                seed = np.random.randint(4, size=6)
                choice = np.random.choice(2, 3)
                img0, mask0 = random_rotation_pair(img0, mask0, random_degree)
                img0, mask0 = random_translation_pair(img0, mask0, seed, choice)
        # ===============random parameters: the sequence should be same ==============#
        
        
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
        img0 = img0 * 255.0
        

        x = np.zeros((2 * self.num_patch * self.num_slice, self.sample_size, self.sample_size))
        for i in range(len(boxes)):
            box = boxes[i].astype('int')[1:]
            begin = max(0, box[0] - self.num_slice // 2)
            end = min(img0.shape[0], box[0] + self.num_slice - self.num_slice // 2)
            tmp_img = img0[begin: end]
            tmp_mask = mask0[begin: end]
            #tmp_patch = tmp_img * 0.5 + tmp_mask * 0.5
            for j in range(len(tmp_img)):
                x[self.num_slice * 2 * i + j] = transform.resize(tmp_img[j], [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
            for j in range(len(tmp_mask)):
                x[self.num_slice * (2 * i + 1)+ j ] = transform.resize(tmp_mask[j], [self.sample_size, self.sample_size], mode='edge', preserve_range='True')
      
        
        x = x.astype('float32')
        return x, y, path.split('/')[-1]         
       
    
class Patch_loader(data.Dataset):
    '''
    Refer to data_tool.py/get_nodule_batch, write on 11/11.
    '''
    
    def __init__(self, list_IDs, dict_paths, labels, pbb_roots, sample_size, num_patch, argu = False, with_global = True, zeropad_patch = True, use_mask = True):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.pbb_roots = pbb_roots
        self.argu = argu
        self.sample_size = sample_size
        self.dict_paths = dict_paths
        self.num_patch = num_patch
        self.with_global = with_global
        self.zeropad_patch = zeropad_patch
        self.use_mask = use_mask

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        def adjust(img_shape, box, ndle_size):
            '''
            img_shape: like (1, 312, 213, 234)
            box: like [23, 45, 54]
            '''
#            print (img_shape, box)
            for i in range(3):         # 3 means three dimensions
                box[i] = max(ndle_size//2, box[i])
                box[i] = min(box[i], img_shape[i] - ndle_size // 2)
            return box
        ID =self.list_IDs[index]
        path = sorted(self.dict_paths[ID])[-1] 

        img_file = path
        
#       pbb = np.load(os.path.join(self.pbb_roots, path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        for pbb_root in self.pbb_roots:
            try:
                pbb = np.load(os.path.join(pbb_root, path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
            except:
                continue
        #pbb = np.load(pbb)
        
        img_3d = nib.load(img_file)
        img0 = img_3d.get_data()
 #       print ('img0.shape: ', img0.shape)
        if self.use_mask:
            mask_file = path.replace("img","mask") 
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            mask0 = mask0 * 255.0
        
        pbb = pbb[pbb[:,0]>-1]
        pbb = nms(pbb,0.05)                                     # used to be 0.05
        print (len(pbb), ' len of pbb')
        #print (pbb)
        if not self.zeropad_patch:
            if (len(pbb) == 0):
                pbb = np.array([[0.5, 0.65 * img0.shape[0], 0.65 * img0.shape[1], 0.65 * img0.shape[2], 64],
                               [0.5, 0.35 * img0.shape[0], 0.65 * img0.shape[1], 0.65 * img0.shape[2], 64],
                               [0.5, 0.65 * img0.shape[0], 0.35 * img0.shape[1], 0.65 * img0.shape[2], 64],
                               [0.5, 0.35 * img0.shape[0], 0.65 * img0.shape[1], 0.35 * img0.shape[2], 64],
                               [0.5, 0.65 * img0.shape[0], 0.35 * img0.shape[1], 0.35 * img0.shape[2], 64]])
            #print (self.num_patch, 'num patch')    
            while len(pbb) < self.num_patch:
                pbb = np.vstack([pbb[0], pbb])
        boxes = pbb[:self.num_patch]                         # 5 means we select the most like nodule region
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-30, 30)   # use to be 20, 20190424
        seed = np.random.randint(10, size=6)  # use to be 4, 20190424
        choice = np.random.choice(2, 3)
        rand = random.uniform(0, 1) 
        
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img0 = img0 * 255.0
        
        channels = self.num_patch
        if self.with_global:
            channels += 1
        if self.use_mask:
            channels += 1
        
        x = 170 * np.ones((channels, self.sample_size, self.sample_size, self.sample_size))
        ndle_size = self.sample_size
        for i in range(len(boxes)):         #self.num_patch           #please be careful about the img and img0
            box = boxes[i].astype('int')[1:]
            #box = adjust(img0.shape, box, ndle_size)
#             x_begin, x_end = box[0] - ndle_size // 2, box[0] + ndle_size//2 
#             y_begin, y_end = box[1] - ndle_size // 2, box[1] + ndle_size//2 
#             z_begin, z_end = box[2] - ndle_size // 2, box[2] + ndle_size//2 
            x_begin, x_end = max(0, box[0] - ndle_size // 2), min(img0.shape[0], box[0] + ndle_size//2)
            y_begin, y_end = max(0, box[1] - ndle_size // 2), min(img0.shape[1], box[1] + ndle_size//2) 
            z_begin, z_end = max(0, box[2] - ndle_size // 2), min(img0.shape[2], box[2] + ndle_size//2)
            
            x_b, x_e = max(0, - box[0] + ndle_size // 2), min(ndle_size, img0.shape[0] - box[0] + ndle_size//2)
            y_b, y_e = max(0, - box[1] + ndle_size // 2), min(ndle_size, img0.shape[1] - box[1] + ndle_size//2)
            z_b, z_e = max(0, - box[2] + ndle_size // 2), min(ndle_size, img0.shape[2] - box[2] + ndle_size//2)
            
            try:
                x[i, x_b:x_e, y_b:y_e, z_b:z_e] = img0[x_begin: x_end, y_begin: y_end, z_begin:z_end]
            except:
                print (x_begin, x_end, y_begin, y_end,z_begin, z_end, img0.shape, ID, box)
                print (x_b, x_e, y_b, y_e,z_b, z_e)
                continue
            
        if self.with_global:
            new_shape = [self.sample_size, self.sample_size, self.sample_size]
            if self.use_mask:
                img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
                mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
                x[-2, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img
                x[-1, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = mask
            else:
                img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
                x[-1, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img
                
        if self.argu and (0 <= rand < 0.8):
            #print ("data augmentation params for train: random_degree, seed, choice: ", random_degree, seed, choice)
            for k in range(len(x)):
                #img, mask = random_rotation_pair(img, mask, random_degree)
                #img, mask = random_translation_pair(img, mask, seed, choice)
                x[k] = random_rotation_single(x[k], random_degree)
                x[k] = random_translation_single(x[k], seed, choice)
        
#        img = np.transpose(img, (2, 0, 1))
#        mask = np.transpose(mask, (2, 0, 1))
        x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype('float32')
        y = self.labels[ID]

        return x, y, ID
    
class Patch_loader2(data.Dataset):    # new patch method, don't need detect nodule. 
    def __init__(self, list_IDs, dict_paths, labels,sample_size, argu = True, use_mask = False):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size
        self.use_mask = use_mask
        self.argu = argu
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        ID =self.list_IDs[index]
        
        path = sorted(self.dict_paths[ID])[-1] 

        img_file = path
        mask_file = path.replace("img","mask") 
        
        img_3d = nib.load(img_file)
        img0 = img_3d.get_data()
        mask_3d = nib.load(mask_file)
        mask0 = mask_3d.get_data()

        whole_shape =  [2 *self.sample_size, 2 *self.sample_size, 2 *self.sample_size]
        
        sample_shape = [self.sample_size, self.sample_size, self.sample_size]
        
        img = transform.resize(img0, whole_shape, mode='edge', preserve_range='True')
        mask = transform.resize(mask0, whole_shape, mode='edge', preserve_range='True')

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        img = (img - img.min()) / (img.max() - img.min())

        if self.argu:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.4):
                random_degree = random.uniform(-15, 15)
                seed = np.random.randint(4, size=6)
                choice = np.random.choice(2, 3)
                img, mask = random_rotation_pair(img, mask, random_degree)
                img, mask = random_translation_pair(img, mask, seed, choice)

 #       img = img * 255.0
 #       mask = mask * 255.0
        if self.use_mask == True:
            x = np.zeros((18, self.sample_size, self.sample_size, self.sample_size))
            for i in range(8):
                x_begin, x_end = (i % 2) * self.sample_size, (i%2 + 1)*self.sample_size
                y_begin, y_end = (i % 4 // 2) * self.sample_size, (i %4 // 2 + 1)*self.sample_size
                z_begin, z_end = (i // 4) * self.sample_size, (i // 4 + 1)*self.sample_size
           #     print (i,x_begin, x_end, y_begin, y_end, z_begin, z_end, img.shape)
                x[2*i] = img[x_begin: x_end, y_begin: y_end, z_begin: z_end]
                x[2 * i + 1] = mask[x_begin: x_end, y_begin: y_end, z_begin: z_end]
        else:
            x = np.zeros((10, self.sample_size, self.sample_size, self.sample_size))
            for i in range(8):
                x_begin, x_end = (i % 2) * self.sample_size, (i%2 + 1)*self.sample_size
                y_begin, y_end = (i % 4 // 2) * self.sample_size, (i %4 // 2 + 1)*self.sample_size
                z_begin, z_end = (i // 4) * self.sample_size, (i // 4 + 1)*self.sample_size
           #     print (i,x_begin, x_end, y_begin, y_end, z_begin, z_end, img.shape)
                x[i] = img[x_begin: x_end, y_begin: y_end, z_begin: z_end]

        x[-2] = transform.resize(img, sample_shape, mode='edge', preserve_range='True')
        x[-1] = transform.resize(mask, sample_shape, mode='edge', preserve_range='True')
        
        x = x.astype('float32')
        y = self.labels[ID]

        return x, y, ID

class MP_Patch_loader(data.Dataset):
    '''
    Refer to data_tool.py/get_nodule_batch, write on 11/11.
    '''
    
    def __init__(self, list_IDs, dict_paths, labels, pbb_roots, global_size, sample_size, num_patch, argu = True, with_global = True, zeropad_patch = True, use_mask = False):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.pbb_roots = pbb_roots
        self.argu = argu
        self.global_size = global_size
        self.sample_size = sample_size
        self.dict_paths = dict_paths
        self.num_patch = num_patch
        self.with_global = with_global
        self.zeropad_patch = zeropad_patch
        self.use_mask = use_mask

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        def adjust(img_shape, box, ndle_size):
            '''
            img_shape: like (1, 312, 213, 234)
            box: like [23, 45, 54]
            '''

            for i in range(3):         # 3 means three dimensions
                box[i] = max(ndle_size//2, box[i])
                box[i] = min(box[i], img_shape[i] - ndle_size // 2)
            return box
        ID =self.list_IDs[index]
        path = sorted(self.dict_paths[ID])[-1] 

        img_file = path
        
#        pbb = np.load(os.path.join(self.pbb_roots, path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        try:
            pbb = np.load(os.path.join(self.pbb_roots[0], path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        except:
            pbb = np.load(os.path.join(self.pbb_roots[1], path.split('/')[-1].replace('.nii.gz', '_pbb.npy')))
#        pbb = np.load(pbb_file)
        
        img_3d = nib.load(img_file)
        img0 = img_3d.get_data()
        if len(img0.shape) == 4:
            #print ('---------', img0.shape)
            img0 = img0[0]
 #       print ('img0.shape: ', img0.shape)
        if self.use_mask:
            mask_file = path.replace("img","mask") 
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            mask0 = mask0 * 255.0
        
        pbb = pbb[pbb[:,0]>-1]
        pbb = nms(pbb,0.01)                                     # used to be 0.05
#         print (len(pbb), ' len of pbb')
        #print (pbb)
        if not self.zeropad_patch:
            if (len(pbb) == 0):
                pbb = np.array([[0.5, 0.65 * img0.shape[0], 0.65 * img0.shape[1], 0.65 * img0.shape[2], 64],
                               [0.5, 0.35 * img0.shape[0], 0.65 * img0.shape[1], 0.65 * img0.shape[2], 64],
                               [0.5, 0.65 * img0.shape[0], 0.35 * img0.shape[1], 0.65 * img0.shape[2], 64],
                               [0.5, 0.35 * img0.shape[0], 0.65 * img0.shape[1], 0.35 * img0.shape[2], 64],
                               [0.5, 0.65 * img0.shape[0], 0.35 * img0.shape[1], 0.35 * img0.shape[2], 64]])
            #print (self.num_patch, 'num patch')    
            while len(pbb) < self.num_patch:
                pbb = np.vstack([pbb[0], pbb])
        boxes = pbb[:self.num_patch]                         # 5 means we select the most like nodule region
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-30, 30)   # use to be 20, 20190424
        seed = np.random.randint(10, size=6)  # use to be 4, 20190424
        choice = np.random.choice(2, 3)
        rand = random.uniform(0, 1)
        
        
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.00001)
        img0 = img0 * 255.0
        
        channels = self.num_patch
        if self.with_global:
            #print ('use global')
            channels += 1
        if self.use_mask:
            #print ('use mask')
            channels += 1
        
        x = np.zeros((self.num_patch, self.sample_size, self.sample_size, self.sample_size))
        x_g = np.zeros((channels - self.num_patch, self.global_size, self.global_size,self.global_size))
        ndle_size = self.sample_size
        #print ('x_g.shape', x_g.shape)
        for i in range(len(boxes)):         #self.num_patch           #please be careful about the img and img0
            box = boxes[i].astype('int')[1:]
            #print (box, img0.shape)
            box = adjust(img0.shape, box, ndle_size)
            #print (box)
            x_begin, x_end = box[0] - ndle_size // 2, box[0] + ndle_size//2 
            y_begin, y_end = box[1] - ndle_size // 2, box[1] + ndle_size//2 
            z_begin, z_end = box[2] - ndle_size // 2, box[2] + ndle_size//2 
            #print (x_begin, x_end, y_begin, y_end,z_begin, z_end, img0.shape, ID)
            x[i, :, :, :] = img0[x_begin: x_end, y_begin: y_end, z_begin:z_end]
            
        if self.with_global:
            new_shape = [self.global_size, self.global_size, self.global_size]
            if self.use_mask:
                img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
                mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
                x_g[-2, 0:self.global_size, 0:self.global_size, 0:self.global_size] = img
                x_g[-1, 0:self.global_size, 0:self.global_size, 0:self.global_size] = mask
            else:
                img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
                x_g[-1, 0:self.global_size, 0:self.global_size, 0:self.global_size] = img
                
        if self.argu and (0 <= rand < 0.8):
            #print ("data augmentation params for train: random_degree, seed, choice: ", random_degree, seed, choice)
            for k in range(len(x)):
                x[k] = random_rotation_single(x[k], random_degree)
                x[k] = random_translation_single(x[k], seed, choice)
            for k in range(len(x_g)):
                x_g[k] = random_rotation_single(x_g[k], random_degree)
                x_g[k] = random_translation_single(x_g[k], seed, choice)
        
#        img = np.transpose(img, (2, 0, 1))
#        mask = np.transpose(mask, (2, 0, 1))
        x = np.transpose(x, (0, 3, 1, 2))
        x_g = np.transpose(x_g, (0, 3, 1, 2))
        x = x.astype('float32')
        x_g = x_g.astype('float32')
        y = self.labels[ID]

        return (x_g, x), y, ID
    


        

