import os,sys

software = 'slicesdir'

command = software

count = 0  


# this part QA for the folder has three subfolders (./subjs/sesses/item)

data_root = "/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/MCL/MCL_time/for_aneri"

subjs = os.listdir(data_root)
fnsh_list = []
empty_list = []
for i in range( len(subjs)):
    sesses = os.listdir(data_root + '/' + subjs[i])
    for j in range(len(sesses)):
        sess_path = data_root + '/' + subjs[i] + '/' + sesses[j]
        items = os.listdir(sess_path)
        if len(items) == 0:
            empty_list.append(sess_path)
        for k in range(len(items)):
            command += ' ' + sess_path + '/' + items[k]
            count += 1
#print(command)
print(count)
print (empty_list)
os.system(command)
'''

data_root = '/share3/gaor2/share5backup/data/NLST/NLSTnorm/reg/img'

imglist = os.listdir(data_root)

for i in range(len(imglist)):
    if i < 0: continue
    if i > 200: break
    print (i, imglist[i])
    command += ' ' + os.path.join(data_root, imglist[i])

os.system(command)
print ('the total precossed image is: ', len(imglist))

'''
