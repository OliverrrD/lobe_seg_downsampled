import os

prep_root = '/nfs/masi/NLST/DSB_File/diag_second/prep'

data_list = os.listdir(prep_root)

save_root = '/nfs/masi/NLST/DSB_File/diag_second/dummy_spread'

print (len(data_list))

for i in range(len(data_list)):
    if data_list[i][-10: ] == '_clean.npy':
        #print (i)
        os.system("mkdir " + save_root + '/' + data_list[i].replace('_clean.npy', ''))

