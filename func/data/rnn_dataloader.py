import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import random
import skimage as sk
from skimage import transform, util
from func.data.data_tool import nms
from func.data.image_manipulation import *

class RNNDataset(data.Dataset):             
    def __init__(self, list_IDs, dict_paths, labels, max_step, sample_size, argu = False, pad = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.output_x = sample_size
        self.output_y = sample_size
        self.output_z = sample_size
        self.img_x = sample_size
        self.img_y = sample_size
        self.img_z = sample_size
        self.max_step = max_step
        self.pad = pad
        self.argu = argu
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        ID =self.list_IDs[index]
        paths = self.dict_paths[ID]  
        paths = sorted(paths)
        x = np.zeros((self.max_step, 2, self.img_z, self.img_x, self.img_y))      
        l = len(paths)
        if l > self.max_step:                     # those have very long sequence, we choose the recent ones.
            paths = paths[-self.max_step:]  
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-15, 15)  
        seed = np.random.randint(4, size=6)
        choice = np.random.choice(2, 3)
        
        for i in range(self.max_step):
            
            path = paths[i % l]
            if i >= l and self.pad:   # for those shorter than 3, we replicate the last one      
                path = paths[-1]
                    
            img_file = path
            mask_file = path.replace("img","mask") # the original one is ".nii.gz"
        
            img_3d = nib.load(img_file)
            img0 = img_3d.get_data()
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()

            new_shape = [self.img_z, self.img_x, self.img_y]
            img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
            mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')

            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            img = (img - img.min()) / (img.max() - img.min())
            
            if self.argu:
                rand = random.uniform(0, 1)
                if (0 <= rand < 0.4):
                    
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)

            img = img * 255.0
            mask = mask * 255.0
            
            x[i, 0, 0:self.output_z, 0:self.output_x, 0:self.output_y] = img[0:self.output_z, 0:self.output_x, 0:self.output_y]
            x[i, 1, 0:self.output_z, 0:self.output_x, 0:self.output_y] = mask[0:self.output_z, 0:self.output_x, 0:self.output_y]
        x = x.astype('float32')
        y = self.labels[ID]
        
        return x, 0, y, ID
    
class DisRNNDataset(data.Dataset):             
    def __init__(self, list_IDs, dict_paths, dict_dists, labels, max_step, sample_size, argu = False, pad = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.dict_dists = dict_dists
        self.output_x = sample_size
        self.output_y = sample_size
        self.output_z = sample_size
        self.img_x = sample_size
        self.img_y = sample_size
        self.img_z = sample_size
        self.max_step = max_step
        self.pad = pad
        self.argu = argu
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        ID =self.list_IDs[index]
        paths = self.dict_paths[ID]  
        paths = sorted(paths)
        points = sorted(self.dict_dists[ID])
        points = [max(points) - i for i in points]
        
        x = np.zeros((self.max_step, 2, self.img_z, self.img_x, self.img_y))       # 2 means 2 channel
        l = len(paths)
        if l > self.max_step:                     # those have very long sequence, we choose the recent ones.
            paths = paths[-self.max_step:]  
            points = points[-self.max_step:]
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-15, 15)  
        seed = np.random.randint(4, size=6)
        choice = np.random.choice(2, 3)
        
        for i in range(self.max_step):
            
            path = paths[i % l]
            
            if i >= l and self.pad:   # for those shorter than 3, we replicate the last one      
                path = paths[-1]
                points.append(0)
                    
            img_file = path
            mask_file = path.replace("img","mask") # the original one is ".nii.gz"
        
            img_3d = nib.load(img_file)
            img0 = img_3d.get_data()
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()

            new_shape = [self.img_z, self.img_x, self.img_y]
            img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
            mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')

            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            img = (img - img.min()) / (img.max() - img.min())
            
            if self.argu:
                rand = random.uniform(0, 1)
                if (0 <= rand < 0.4):
                    
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)

            img = img * 255.0
            mask = mask * 255.0
            
            x[i, 0, 0:self.output_z, 0:self.output_x, 0:self.output_y] = img[0:self.output_z, 0:self.output_x, 0:self.output_y]
            x[i, 1, 0:self.output_z, 0:self.output_x, 0:self.output_y] = mask[0:self.output_z, 0:self.output_x, 0:self.output_y]
        x = x.astype('float32')
        y = self.labels[ID]
        
        points = np.array(points, dtype = 'float32')
        
        
        return x, points, y, ID


class RNNMultiS_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_labels, max_step, sample_size, num_patch, argu=False, lastscan=False,
                 use_mask=True, ave=False, threed=False, argu_3d=False, pad = True):
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
        self.max_step = max_step
        self.pad = pad

    def get_all_IDs(self):
        all_paths = []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)
        return all_paths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        paths = self.dict_paths[ID]
        paths = sorted(paths)

        l = len(paths)
        if l > self.max_step:
            paths = paths[-self.max_step:]
        #print ('paths', paths)
        random_degree = random.uniform(-30, 30)
        seed = np.random.randint(4, size=4)
        choice = np.random.choice(2, 2)
        rand = random.uniform(0, 1)
        channels = self.num_patch
        if self.ave:
            x = 170 * np.zeros((self.max_step, channels, self.sample_size, self.sample_size), dtype=np.uint8)
        else:
            x = 170 * np.zeros((self.max_step, 2 * channels, self.sample_size, self.sample_size), dtype=np.uint8)

        for i in range(self.max_step):
            path = paths[i % l]
            if i >= l and self.pad:
                path = paths[-1]

            img_file = path

            #try:
            img_3d = nib.load(img_file)
            #except:
             #   print(img_file, ' has problem !')
            img0 = img_3d.get_data()
        #       print ('img0.shape: ', img0.shape)
        #if self.use_mask:
            mask_file = path.replace("img", "mask")
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0
            if self.argu:
                if (0 <= rand < 0.6):
                    if self.use_mask:
                        img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                        img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)

            img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
            img0 = img0 * 255.0

            for j in range(channels // 5):  # self.num_patch     #please be careful about the img and img0
                if self.threed:
                    tmp_img = img0[5 * j: 5 * j + 5]  # simulation use i, other may use 3 * i
                    tmp_mask = mask0[5 * j: 5 * j + 5]
                else:
                    tmp_img = img0[15 * j: 15 * j + 5]  # simulation use i, other may use 3 * i
                    tmp_mask = mask0[15 * j: 15 * j + 5]

                if self.ave:

                    x[i, 5 * j: 5 * j + 5] = tmp_img * (0.5 + tmp_mask / 255.0 * 0.5)
                    # x[10*i + 5: 10 * i + 10] = tmp_mask
                else:

                    x[i, 10 * j: 10 * j + 5] = tmp_img
                    x[i, 10 * j + 5: 10 * j + 10] = tmp_mask

            #         if self.argu_3d:
            #             assert x.shape[0] == 18
            #             index1, index2 = np.random.randint(0,3,2)
            #             #index1, index2 = 2, 1
            #             for i in range(3):
            # #                 x[i*6: i * 6 + 6] = self.swap(x[i*6: i * 6 + 6], index1, index2)
        # nii_img = nib.Nifti1Image(x[0].data, affine=np.eye(4))
        # nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + paths[0].split('/')[-1])
        # nii_img = nib.Nifti1Image(x[1].data, affine=np.eye(4))
        # nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + paths[1].split('/')[-1])
            #         x = z
        x = x.astype('float32')
        y = self.path_labels[paths[-1]]
            #print ("x and y: ", x.shape, y)
        return x, y, path.split('/')[-1]



class RNNDiagMultiS_REG_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, dict_diags, path_labels, max_step, sample_size, num_patch, argu=False, lastscan=False,
                 use_mask=True, ave=False, threed=False, argu_3d=False, pad = True):
        'Initialization'
        self.path_labels = path_labels
        self.dict_diags = dict_diags
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
        self.max_step = max_step
        self.pad = pad

    def get_all_IDs(self):
        all_paths = []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)
        return all_paths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        paths = self.dict_paths[ID]
        paths = sorted(paths)
        diags = self.dict_diags[ID]
        diags = sorted(diags, reverse=True)

        l = len(paths)
        if l > self.max_step:
            paths = paths[-self.max_step:]
            diags = diags[-self.max_step:]

        #print ('paths', paths)
        random_degree = random.uniform(-40, 40)
        seed = np.random.randint(10, size=4)
        choice = np.random.choice(2, 2)
        rand = random.uniform(0, 1)
        channels = self.num_patch
        if self.ave:
            x = 170 * np.zeros((self.max_step, channels, self.sample_size[0], self.sample_size[1]), dtype=np.uint8)
        else:
            x = 170 * np.zeros((self.max_step, 2 * channels, self.sample_size[0], self.sample_size[1]), dtype=np.uint8)
        diag_vec = np.zeros(self.max_step, dtype = np.float)
        for i in range(self.max_step):
            path = paths[i % l]
            diag = diags[i % l]
            if i >= l and self.pad:
                path = paths[-1]
                diag = diags[-1]
            diag_vec[i] = diag
            img_file = path


            img_3d = nib.load(img_file)

            img0 = img_3d.get_data()

            mask_file = path.replace("img", "mask")
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0
            if self.argu:
                if (0 <= rand < 0.8):
                    if self.use_mask:
                        img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                        img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)

            img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
            img0 = img0 * 255.0

            for j in range(channels // 5):  # self.num_patch     #please be careful about the img and img0

                tmp_img = img0[5 * j: 5 * j + 5]  # simulation use i, other may use 3 * i
                tmp_mask = mask0[5 * j: 5 * j + 5]

                if self.ave:

                    x[i, 5 * j: 5 * j + 5] = tmp_img * (0.5 + tmp_mask / 255.0 * 0.5)
                else:
                    x[i, 10 * j: 10 * j + 5] = tmp_img
                    x[i, 10 * j + 5: 10 * j + 10] = tmp_mask

        x = x.astype('float32')
        diag_vec = diag_vec.astype('float32')
        y = self.path_labels[paths[-1]]
            #print ("x and y: ", x.shape, y)
        return x, diag_vec, y, path.split('/')[-1]

class RNNDiagMultiS_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, dict_diags, path_labels, max_step, sample_size, num_patch, argu=False, lastscan=False,
                 use_mask=True, ave=False, threed=False, argu_3d=False, pad = True):
        'Initialization'
        self.path_labels = path_labels
        self.dict_diags = dict_diags
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
        self.max_step = max_step
        self.pad = pad

    def get_all_IDs(self):
        all_paths = []
        for ID in self.list_IDs:
            tmp_list = self.dict_paths[ID]
            for path in tmp_list:
                all_paths.append(path)
        return all_paths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        paths = self.dict_paths[ID]
        paths = sorted(paths)
        diags = self.dict_diags[ID]
        diags = sorted(diags, reverse=True)

        l = len(paths)
        if l > self.max_step:
            paths = paths[-self.max_step:]
            diags = diags[-self.max_step:]

        #print ('paths', paths)
        random_degree = random.uniform(-30, 30)
        seed = np.random.randint(4, size=4)
        choice = np.random.choice(2, 2)
        rand = random.uniform(0, 1)
        channels = self.num_patch
        if self.ave:
            x = 170 * np.zeros((self.max_step, channels, self.sample_size, self.sample_size), dtype=np.uint8)
        else:
            x = 170 * np.zeros((self.max_step, 2 * channels, self.sample_size, self.sample_size), dtype=np.uint8)
        diag_vec = np.zeros(self.max_step, dtype = np.float)
        for i in range(self.max_step):
            path = paths[i % l]
            diag = diags[i % l]
            if i >= l and self.pad:
                path = paths[-1]
                diag = diags[-1]
            diag_vec[i] = diag
            img_file = path


            img_3d = nib.load(img_file)

            img0 = img_3d.get_data()

            mask_file = path.replace("img", "mask")
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            if np.max(mask0) < 2:
                mask0 = mask0 * 255.0
            if self.argu:
                if (0 <= rand < 0.6):
                    if self.use_mask:
                        img0, mask0 = random_rotation_pair2d(img0, mask0, random_degree)
                        img0, mask0 = random_translation_pair2d(img0, mask0, seed, choice)

            img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 0.0001)
            img0 = img0 * 255.0

            for j in range(channels // 5):  # self.num_patch     #please be careful about the img and img0
                if self.threed:
                    tmp_img = img0[5 * j: 5 * j + 5]  # simulation use i, other may use 3 * i
                    tmp_mask = mask0[5 * j: 5 * j + 5]
                else:
                    tmp_img = img0[15 * j: 15 * j + 5]  # simulation use i, other may use 3 * i
                    tmp_mask = mask0[15 * j: 15 * j + 5]

                if self.ave:

                    x[i, 5 * j: 5 * j + 5] = tmp_img * (0.5 + tmp_mask / 255.0 * 0.5)

                else:

                    x[i, 10 * j: 10 * j + 5] = tmp_img
                    x[i, 10 * j + 5: 10 * j + 10] = tmp_mask

        # nii_img = nib.Nifti1Image(x[0].data, affine=np.eye(4))
        # nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + paths[0].split('/')[-1])
        # nii_img = nib.Nifti1Image(x[1].data, affine=np.eye(4))
        # nib.save(nii_img, '/nfs/masi/gaor2/tmp/test/' + paths[1].split('/')[-1])
            #         x = z
        x = x.astype('float32')
        diag_vec = diag_vec.astype('float32')
        y = self.path_labels[paths[-1]]
            #print ("x and y: ", x.shape, y)
        return x, diag_vec, y, path.split('/')[-1]


class DisRnnPatch_loader(data.Dataset):         
    
    def __init__(self, list_IDs, dict_paths, dict_dists, labels, pbb_roots, max_step, sample_size, num_patch, argu = False, pad = True, with_global = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size
        self.pbb_roots = pbb_roots
        self.max_step = max_step
        self.pad = pad
        self.argu =argu
        self.num_patch = num_patch
        self.with_global = with_global
        self.dict_dists = dict_dists

        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        ID =self.list_IDs[index]
        paths = sorted(self.dict_paths[ID]) 
        points = sorted(self.dict_dists[ID])
        points = [max(points) - i for i in points]
        
        if self.with_global:
            x = np.zeros((self.max_step, self.num_patch + 2, self.sample_size, self.sample_size, self.sample_size))       # 3 means 3 time step
        else:
            x = np.zeros((self.max_step, self.num_patch, self.sample_size, self.sample_size, self.sample_size))  
            
        l = len(paths)
        last_img_3d = nib.load(paths[-1])
        last_img = last_img_3d.get_data()
        if l > self.max_step:                     # those have very long sequence, we choose the recent ones.
            paths = paths[-self.max_step:] 
            points = points[-self.max_step:]
        
        #print (paths[-1])
        #print (paths[-1].split('/')[-1])
        pbb = np.load(os.path.join(self.pbb_roots, paths[-1].split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        
            
        pbb = pbb[pbb[:,0]>-1]
        pbb = nms(pbb,0.01)                 # use
        if (len(pbb) == 0):
            pbb = np.array([[0.5, 0.65 * last_img.shape[0], 0.65 * last_img.shape[1], 0.65 * last_img.shape[2], 64],
                           [0.5, 0.35 * last_img.shape[0], 0.65 * last_img.shape[1], 0.65 * last_img.shape[2], 64],
                           [0.5, 0.65 * last_img.shape[0], 0.35 * last_img.shape[1], 0.65 * last_img.shape[2], 64],
                           [0.5, 0.35 * last_img.shape[0], 0.65 * last_img.shape[1], 0.35 * last_img.shape[2], 64],
                           [0.5, 0.65 * last_img.shape[0], 0.35 * last_img.shape[1], 0.35 * last_img.shape[2], 64]])
     
        while len(pbb) < self.num_patch:
            pbb = np.vstack([pbb[0], pbb])
        boxes = pbb[:self.num_patch]
        #print ('boxes', boxes)
        ndle_size = self.sample_size
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-15, 15)  
        seed = np.random.randint(4, size=6)
        choice = np.random.choice(2, 3)
        rand = random.uniform(0, 1)
        
        for j in range(len(boxes)):
            boxes[j] = boxes[j].astype('int')   
            boxes[j][1:] = adjust(last_img.shape, boxes[j][1:], ndle_size)
            
        for i in range(self.max_step):
            if i >= l and self.pad:   # for those shorter than 3, we replicate the last one      
                path = paths[-1]
                points.append(0)
            path = paths[i % l]
            img_file = path
            mask_file = path.replace("img","mask") # the original one is ".nii.gz"
            
            img_3d = nib.load(img_file)
            img0 = img_3d.get_data()
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            
            img0 = (img0 - img0.min()) / (img0.max() - img0.min())
            img0 = img0 * 255.0
            mask0 = mask0 * 255.0
            
            for j in range(self.num_patch):                    #please be careful about the img and img0
                box = boxes[j][1:]
#                print ('box',box, j)
                x_center, y_center, z_center = int(box[0] * img0.shape[0]), int(box[1] * img0.shape[1]), int (box[2] * img0.shape[2])
                x_center = min(max(ndle_size//2, x_center), img0.shape[0] - ndle_size // 2)
                y_center = min(max(ndle_size//2, y_center), img0.shape[1] - ndle_size // 2)
                z_center = min(max(ndle_size//2, z_center), img0.shape[2] - ndle_size // 2)
                #print ('img0.shape', img0.shape)
                #print ('x, y, z center, ndle_size', x_center, y_center, z_center, ndle_size)
                x[i, j, :, :, :] = img0[ (x_center - ndle_size // 2): (x_center + ndle_size//2), 
                                   (y_center - ndle_size// 2): (y_center + ndle_size//2), 
                                   (z_center - ndle_size//2): (z_center + ndle_size//2)]
            
            if self.with_global:
                new_shape = [self.sample_size, self.sample_size, self.sample_size]
                img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
                mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
                x[i, -2, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img[0:self.sample_size, 0:self.sample_size, 0:self.sample_size]
                x[i, -1, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = mask[0:self.sample_size, 0:self.sample_size, 0:self.sample_size]


            if self.argu and (0 <= rand < 0.4):
                for k in range(len(x[i])):
                    x[i][k] = random_rotation_single(x[i][k], random_degree)
                    x[i][k] = random_translation_single(x[i][k], seed, choice)
            
        x = np.transpose(x, (0, 1, 4, 2, 3))
        x = x.astype('float32')
        y = self.labels[ID]
        #print (points)
        points = np.array(points, dtype = 'float32')
        
        
        return x, points, y, ID
    
    
class RnnPatch_loader(data.Dataset):         
    
    def __init__(self, list_IDs, dict_paths, labels, pbb_roots, max_step, sample_size, num_patch, argu = False, pad = True, zeropad_patch = True, with_global = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size
        self.pbb_roots = pbb_roots
        self.max_step = max_step
        self.pad = pad
        self.argu =argu
        self.num_patch = num_patch
        self.with_global = with_global
        self.zeropad_patch = zeropad_patch

        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        ID =self.list_IDs[index]
        paths = sorted(self.dict_paths[ID])
        
        
        if self.with_global:
            x = np.zeros((self.max_step, self.num_patch + 2, self.sample_size, self.sample_size, self.sample_size))       # 3 means 3 time step
        else:
            x = np.zeros((self.max_step, self.num_patch, self.sample_size, self.sample_size, self.sample_size))  
            
        l = len(paths)
        last_img_3d = nib.load(paths[-1])
        last_img = last_img_3d.get_data()
        if l > self.max_step:                     # those have very long sequence, we choose the recent ones.
            paths = paths[-self.max_step:]  
        
        #print (paths[-1])
        #print (paths[-1].split('/')[-1])
        pbb = np.load(os.path.join(self.pbb_roots, paths[-1].split('/')[-1].replace('.nii.gz', '_pbb.npy')))
        
            
        pbb = pbb[pbb[:,0]>-1]
        pbb = nms(pbb,0.05)                 # use
        if not self.zeropad_patch:
            if (len(pbb) == 0):
                pbb = np.array([[0.5, 0.65 * last_img.shape[0], 0.65 * last_img.shape[1], 0.65 * last_img.shape[2], 64],
                               [0.5, 0.35 * last_img.shape[0], 0.65 * last_img.shape[1], 0.65 * last_img.shape[2], 64],
                               [0.5, 0.65 * last_img.shape[0], 0.35 * last_img.shape[1], 0.65 * last_img.shape[2], 64],
                               [0.5, 0.35 * last_img.shape[0], 0.65 * last_img.shape[1], 0.35 * last_img.shape[2], 64],
                               [0.5, 0.65 * last_img.shape[0], 0.35 * last_img.shape[1], 0.35 * last_img.shape[2], 64]])

            while len(pbb) < self.num_patch:
                pbb = np.vstack([pbb[0], pbb])
        boxes = pbb[:self.num_patch]
        #print ('boxes', boxes)
        ndle_size = self.sample_size
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-30, 30)  
        seed = np.random.randint(10, size=6)
        choice = np.random.choice(2, 3)
        rand = random.uniform(0, 1)
        
        for j in range(len(boxes)):
            boxes[j] = boxes[j].astype('int')   
            boxes[j][1:] = adjust(last_img.shape, boxes[j][1:], ndle_size)
            
        for i in range(self.max_step):
            if i >= l:
                if self.pad:
                    path = paths[-1]
            path = paths[i % l]
            img_file = path
            mask_file = path.replace("img","mask") # the original one is ".nii.gz"
            
            img_3d = nib.load(img_file)
            img0 = img_3d.get_data()
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            
            img0 = (img0 - img0.min()) / (img0.max() - img0.min())
            img0 = img0 * 255.0
            mask0 = mask0 * 255.0
            
            for j in range(len(boxes)):                    #please be careful about the img and img0
                box = boxes[j][1:]
#                print ('box',box, j)
                x_center, y_center, z_center = int(box[0] * img0.shape[0]), int(box[1] * img0.shape[1]), int (box[2] * img0.shape[2])
                x_center = min(max(ndle_size//2, x_center), img0.shape[0] - ndle_size // 2)
                y_center = min(max(ndle_size//2, y_center), img0.shape[1] - ndle_size // 2)
                z_center = min(max(ndle_size//2, z_center), img0.shape[2] - ndle_size // 2)
                #print ('img0.shape', img0.shape)
                #print ('x, y, z center, ndle_size', x_center, y_center, z_center, ndle_size)
                x[i, j, :, :, :] = img0[ (x_center - ndle_size // 2): (x_center + ndle_size//2), 
                                   (y_center - ndle_size// 2): (y_center + ndle_size//2), 
                                   (z_center - ndle_size//2): (z_center + ndle_size//2)]
            
            if self.with_global:
                new_shape = [self.sample_size, self.sample_size, self.sample_size]
                img = transform.resize(img0, new_shape, mode='edge', preserve_range='True')
                mask = transform.resize(mask0, new_shape, mode='edge', preserve_range='True')
                x[i, -2, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img[0:self.sample_size, 0:self.sample_size, 0:self.sample_size]
                x[i, -1, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = mask[0:self.sample_size, 0:self.sample_size, 0:self.sample_size]


            if self.argu and (0 <= rand < 0.8):
                for k in range(len(x[i])):
                    x[i][k] = random_rotation_single(x[i][k], random_degree)
                    x[i][k] = random_translation_single(x[i][k], seed, choice)
            
        x = np.transpose(x, (0, 1, 4, 2, 3))
        x = x.astype('float32')
        y = self.labels[ID]
        
        return x, 0, y, ID    
    
class RnnPatch_loader2(data.Dataset):         
    
    def __init__(self, list_IDs, dict_paths, labels, max_step, sample_size, argu = False, pad = True):   
        # list_IDs: list of id, labels: dict{'id': label}, list_paths: list of list (path)
        self.labels = labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.sample_size = sample_size

        self.max_step = max_step
        self.pad = pad
        self.argu = argu
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        ID =self.list_IDs[index]
        paths = self.dict_paths[ID]  
        x = np.zeros((self.max_step, 10, self.sample_size, self.sample_size, self.sample_size))       # 3 means 3 time step
        l = len(paths)
        if l > self.max_step:                     # those have very long sequence, we choose the recent ones.
            paths = paths[-self.max_step:]  
            
        whole_shape =  [2 *self.sample_size, 2 *self.sample_size, 2 *self.sample_size]
        
        sample_shape = [self.sample_size, self.sample_size, self.sample_size]
        
        # ===============random parameters: the sequence should be same ==============#
        random_degree = random.uniform(-15, 15)  
        seed = np.random.randint(4, size=6)
        choice = np.random.choice(2, 3)
            
        for i in range(self.max_step):
            path = paths[i % l]
            if i >= l:
                if self.pad:
                    path = paths[-1]
            img_file = path
            mask_file = path.replace("img","mask") # the original one is ".nii.gz"
            
            img_3d = nib.load(img_file)
            img0 = img_3d.get_data()
            mask_3d = nib.load(mask_file)
            mask0 = mask_3d.get_data()
            
            img = transform.resize(img0, whole_shape, mode='edge', preserve_range='True')
            mask = transform.resize(mask0, whole_shape, mode='edge', preserve_range='True')
            
            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255.0
            mask = mask * 255.0
            
            if self.argu:
                rand = random.uniform(0, 1)
                if (0 <= rand < 0.4):
                    
                    img, mask = random_rotation_pair(img, mask, random_degree)
                    img, mask = random_translation_pair(img, mask, seed, choice)
            
            for j in range(8):                    #please be careful about the img and img0
                x_begin, x_end = (i % 2) * self.sample_size, (i%2 + 1)*self.sample_size
                y_begin, y_end = (i % 4 // 2) * self.sample_size, (i %4 // 2 + 1)*self.sample_size
                z_begin, z_end = (i // 4) * self.sample_size, (i // 4 + 1)*self.sample_size

                x[i, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = img[x_begin: x_end, y_begin: y_end, z_begin: z_end]


            
            x[i, -2, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = transform.resize(img0, sample_shape, mode='edge', preserve_range='True')
            x[i, -1, 0:self.sample_size, 0:self.sample_size, 0:self.sample_size] = transform.resize(mask0, sample_shape, mode='edge', preserve_range='True')
        x = x.astype('float32')
        y = self.labels[ID]
        
        return x, y, ID    
    

def adjust(img_shape, box, ndle_size):

#            print (img_shape, box)
            for i in range(3):
                box[i] = max(ndle_size//2, box[i])
                box[i] = min(box[i], img_shape[i] - ndle_size // 2)
                box[i] = float(box[i]) / img_shape[i]
            return box

def rnn_forwarder(rnn, inputs, seq_lengths):
    """
    # https://blog.csdn.net/u012436149/article/details/79749409 
    :param rnn: RNN instance
    :param inputs: FloatTensor, shape [batch, time, dim] if rnn.batch_first else [time, batch, dim]
    :param seq_lengths: LongTensor shape [batch]
    :return: the result of rnn layer,
    """
    batch_first = rnn.batch_first

    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)

    _, desorted_indices = torch.sort(indices, descending=False)

    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]
    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs,
                                                      sorted_seq_lengths.cpu().numpy(),
                                                      batch_first=batch_first)

    res, state = rnn(packed_inputs)

    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first)


    if batch_first:
        desorted_res = padded_res[desorted_indices]
    else:
        desorted_res = padded_res[:, desorted_indices]
    return desorted_res

    
def sort_batch(data, label, seq_lengths, batch_first):
    if isinstance(seq_lengths, list):
        seq_lengths = torch.from_numpy(np.array(seq_lengths))
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    if batch_first:
        data = data[indices]
    else: 
        data = data[:, indices]
    label = label[indices]
    packed_data = nn.utils.rnn.pack_padded_sequence(data,
                                                      sorted_seq_lengths.cpu().numpy(),
                                                      batch_first=batch_first)  
    return packed_data, label
        
        