import pyxnat
import dax

xnat = dax.XnatUtils.get_interface()

prearchive = pyxnat.core.manage.PreArchive(xnat)


# Retrieves list
pre_data = prearchive.get()

f = open('/nfs/masi/MCL/file/txt/MCL_prearchive20191217.txt')
lines = f.readlines()
lines = [line.strip() for line in lines]

archive_list = []

for i in range(len(pre_data)):
    data = pre_data[i]
    if data[2] in lines:
        print (data)
        archive_list.append(data)
#print (archive_list, len(archive_list)) 

for i in range(len(archive_list)):
    if i > 0: break
    sess = archive_list[i][2]
    proj = archive_list[i][0]
    subj = sess.split('-')[0]
    date = archive_list[i][1]
    src = '/prearchive/projects/' + proj + '/' + date + '/' + sess
    post_body = """src={src}&project={proj}&subject={subj}&session={sess}&overwrite=delete""".format(
        src=src,
        proj=proj,
        subj=subj,
        sess=sess
        )
    request_uri = '/data/services/archive'
    print (post_body)
    try:
        xnat._exec(
            request_uri, 'POST', post_body,
            {'content-type':'application/x-www-form-urlencoded'}
       )
    except:
        print ('--error---')
        continue

#    break

    
 






