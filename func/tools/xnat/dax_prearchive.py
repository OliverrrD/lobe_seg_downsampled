import dax
from flatten_json import flatten_json
xnat = dax.XnatUtils.get_interface()
prearchive = xnat.get('/data/prearchive/projects/MCL')
json_obj = prearchive.json()
#print (json_obj)
flat_json = flatten_json(json_obj)
keys=flat_json.keys()
names=[]
for i in keys:
    print (i)
    if 'name' in i:
        names.append(i)
f = open('/nfs/masi/MCL/file/txt/MCL_prearchive20200901.txt', 'w')
for name in names:
    print (flat_json[name] + '\n')
    f.write(flat_json[name] + '\n')
f.close()