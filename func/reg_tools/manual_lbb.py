

import os
import numpy as np

bbox_root = '/nfs/masi/MCL/DSB_File/tmp_dsb_file0404/align_bbox'

data_list = os.listdir(bbox_root)

print (len(data_list))

#data_list = [i[:14] for i in data_list]

for i in range(len(data_list)):
	print (bbox_root + '/' + data_list[i].replace('pbb', 'lbb'))


	np.save( bbox_root + '/' + data_list[i].replace('pbb', 'lbb'), np.zeros((1,4), dtype = np.float32))

