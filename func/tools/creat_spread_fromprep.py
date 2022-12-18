import os 

prep_root = '/nfs/masi/gaor2/data/Reg_formedia/resample/ref_DSB/prep'
spread_root = '/nfs/masi/gaor2/data/Reg_formedia/resample/ref_DSB/fake_spread'

data_list = os.listdir(prep_root)

data_list = [i.split('_')[0] for i in data_list]

data_list = list(set(data_list))

for item in data_list:
    os.mkdir(spread_root + '/' + item)