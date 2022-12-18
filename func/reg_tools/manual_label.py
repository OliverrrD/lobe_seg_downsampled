

import os
import numpy as np

prep_root = '/nfs/masi/NLST/registration/20200201_NLST_missing_data_local_config_v4/After_REG/REG_DSB/prep'

data_list = os.listdir(prep_root)

print (len(data_list))

#data_list = [i[:14] for i in data_list]

for i in range(len(data_list)):
	print (prep_root + '/' + data_list[i].replace('_clean.npy', '_label.npy'))
	np.save( prep_root + '/' + data_list[i].replace('_clean.npy', '_label.npy'), np.array([[0, 0, 0, 0]]))

