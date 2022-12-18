import os
import numpy as np
import nibabel as nib
import random
from scipy import ndarray
from skimage import transform, util

def random_rotation(image_array,degree):
    # pick a random degree of rotation between 15% on the left and 15% on the right
    random_degree = random.uniform(-degree, degree)
    return transform.rotate(image_array, random_degree,preserve_range=True)
'''
def random_noise(image_array: ndarray):
    # add random noise to the image
    return util.random_noise(image_array)
'''

#in ZXY representation
def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:,:,::-1]

#in ZXY representation
def vertical_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1,:]

def random_translation(image_array,limit):
    seed=np.random.randint(limit, size=6)
    choice=np.random.choice(2, 3)
    
    rt=util.crop(copy=True, ar=image_array, crop_width=((seed[0],seed[1]),(seed[2],seed[3]),(seed[4],seed[5])))
    
    if(choice[0]==0):
        rt=util.pad(rt,((seed[0]+seed[1],0),(0,0),(0,0)),'constant')
    else:
        rt=util.pad(rt,((0,seed[0]+seed[1]),(0,0),(0,0)),'constant')
        
    if(choice[1]==0):
        rt=util.pad(rt,((0,0),(seed[2]+seed[3],0),(0,0)),'constant')
    else:
        rt=util.pad(rt,((0,0),(0,seed[2]+seed[3]),(0,0)),'constant')
        
    if(choice[2]==0):
        rt=util.pad(rt,((0,0),(0,0),(seed[4]+seed[5],0)),'constant')
    else:
        rt=util.pad(rt,((0,0),(0,0),(0,seed[4]+seed[5])),'constant')
        
    return rt

'''
#augmentation
if(random.uniform(0, 1)>0.7):
rot1,rot2=random.sample(set([0,1,2]), 2)
reshaped=np.rot90(reshaped,axes=(rot1,rot2))
'''

def random_rotation_twoarrays(image_array,mask_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-15, 15)
    rot_img = transform.rotate(image_array, random_degree, preserve_range=True)
    rot_mask = transform.rotate(mask_array, random_degree, preserve_range=True)
    return rot_img,rot_mask

def random_rotation_pair(image, mask, random_degree):  
    rot_img = transform.rotate(image, random_degree, preserve_range=True)
    rot_mask = transform.rotate(mask, random_degree, preserve_range=True)
    return rot_img,rot_mask

def random_rotation_pair2d(image, mask, random_degree):
    assert len(image) == len(mask) and len(image.shape) in [2, 3]
    if len(image.shape) == 3:
        for i in range(len(image)):
            image[i] = transform.rotate(image[i], random_degree, preserve_range = True)
            mask[i] = transform.rotate(mask[i], random_degree, preserve_range = True)
    else:
        image = transform.rotate(image, random_degree, preserve_range=True)
        mask = transform.rotate(mask, random_degree, preserve_range=True)
    return image, mask

def random_translation_pair2d(image, mask, seed, choice):
    #print (len(image), len(mask), len(image.shape))
    assert len(image) == len(mask) 
    for i in range(len(image)):
        image_array = image[i]
        mask_array = mask[i]
        rt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3])))

        if (choice[0] == 0):
            rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0)), 'constant')
        else:
            rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0)), 'constant')

        if (choice[1] == 0):
            rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0)), 'constant')
        else:
            rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3])), 'constant')

        mt = util.crop(copy=True, ar=mask_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3])))

        if (choice[0] == 0):
            mt = util.pad(mt, ((seed[0] + seed[1], 0), (0, 0)), 'constant')
        else:
            mt = util.pad(mt, ((0, seed[0] + seed[1]), (0, 0)), 'constant')

        if (choice[1] == 0):
            mt = util.pad(mt, ((0, 0), (seed[2] + seed[3], 0)), 'constant')
        else:
            mt = util.pad(mt, ((0, 0), (0, seed[2] + seed[3])), 'constant')
        image[i] = rt
        mask[i] = mt

    return image, mask

def random_rotation_single(image, random_degree):
    rot_img = transform.rotate(image, random_degree, preserve_range=True)
    return rot_img

def random_translation_pair(image_array, mask_array, seed, choice):
#    seed = np.random.randint(limit, size=6)
#    choice = np.random.choice(2, 3)

    rt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        rt = util.pad(rt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    mt = util.crop(copy=True, ar=mask_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        mt = util.pad(mt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        mt = util.pad(mt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        mt = util.pad(mt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    return rt, mt

def random_translation_single(image_array,seed, choice):
#    seed = np.random.randint(limit, size=6)
#    choice = np.random.choice(2, 3)

    rt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        rt = util.pad(rt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    return rt

def random_translation_twoarrays(image_array, mask_array, limit):
    seed = np.random.randint(limit, size=6)
    choice = np.random.choice(2, 3)

    rt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        rt = util.pad(rt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    mt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        mt = util.pad(mt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        mt = util.pad(mt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        mt = util.pad(mt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    return rt, mt