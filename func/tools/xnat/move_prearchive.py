import pyxnat
import dax

xnat = dax.XnatUtils.get_interface()

prearchive = pyxnat.core.manage.PreArchive(xnat)

pre_data = prearchive.get()
'''
f = open('/home/local/VANDERBILT/gaor2/Desktop/spore_lost0.txt')
lines = f.readlines()
lines = [line.strip() for line in lines]
'''
lines = ['15595433147-20160215', '2279063907-20180206', '12541965268-20180406', '5434231673-20180223']
move_list = []
cnt = 0
for i in range(len(pre_data)):
    data = pre_data[i]
    if data[2] in lines:
        cnt += 1
        print (data)
        move_list.append(data)
        uri = prearchive.get_uri(data)
        prearchive.move([uri],'SPORE')
        
# 
# print (uri)
