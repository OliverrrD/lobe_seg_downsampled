import numpy as np
import os
import nibabel as nib

def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def get_img_heatmap(pbb_path, clean_path, save_path):
    pbb = np.load(pbb_path)

    clean = np.load(clean_path)

    pbb = pbb[pbb[:,0]>-1.5]
    pbb = nms(pbb,0.05)
    boxes = pbb[:5]
    shape = clean.shape[1:]
    coors = np.mgrid[:shape[0], :shape[1], :shape[2]]
    heat_map = np.zeros(shape)
    assert len(shape) == 3
    for i in range(len(boxes)):
        box = boxes[i][1:]
        assert len(box) == 4  # center_x, center_y, center_z, nodule_size
        distance_map = (coors[0] - box[0]) ** 2 + (coors[1] - box[1]) ** 2 + (coors[2] - box[2]) ** 2
        heat_map +=  np.exp(-distance_map / (4096) * (i + 1)) # here could change 
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
    npy = clean * heat_map
    npy = np.uint8(npy)
    array_img = nib.Nifti1Image(npy, np.eye(4))
    nib.save(array_img, save_path)
    

if __name__ == "__main__":
    pbb_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/DSB_File/noreg/bbox'
    save_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/NLSTnorm/noreg/heatmap4096'
    clean_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/DSB_File/noreg/prep'
    clean_list = os.listdir(clean_root)
    id_list = list(set([i[:-10] for i in clean_list]))  # the operation set will change the order of list
    for i in range(len(id_list)):
        #if i>1: break
        print (i, len(id_list), id_list[i], clean_list[i])
        get_img_heatmap(pbb_root + '/' + id_list[i] + '_pbb.npy', clean_root + '/' +  id_list[i] + '_clean.npy', save_root + '/' +  id_list[i] + '.nii.gz')
    
    
    
    
    