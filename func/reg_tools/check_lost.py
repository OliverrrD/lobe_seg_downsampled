import os
import argparse
import nibabel as nib
from multiprocessing import Pool

def main():
    parser = argparse.ArgumentParser(description='check if the folder contains bad files')
    parser.add_argument('--in_folder', type=str,
                        help='The input image folder')
    parser.add_argument('--out', type=str,
                        help='The output bad list txt file')
    parser.add_argument('--num_processes', type=int, default=20)
    args = parser.parse_args()
    data_list = os.listdir(args.in_folder)
    print ("the folder", args.in_folder, " contains ", len(data_list), ' files')
    pool = Pool(processes=args.num_processes)

    file_name_chunks = get_chunks_list(data_list, args.num_processes)
    sum_list = []
    result_list = [pool.apply_async(get_bad_list_chunk, (file_name_chunk, args.in_folder)) for file_name_chunk in
                   file_name_chunks]

    for tmp in result_list:
        tmp.wait()
        tmp_list = tmp.get()
        sum_list += tmp_list
    print (sum_list)


def get_chunks_list(in_list, num_chunks):
    return [in_list[i::num_chunks] for i in range(num_chunks)]


def get_bad_list_chunk(data_list, in_fold):

    bad_list = []
    for i in range(len(data_list)):
        if i % 5 == 0: print (i , len(data_list))
        try:
            img = nib.load(in_fold + '/' + data_list[i]).get_data()
        except:
            bad_list.append(data_list[i])
    return bad_list

if __name__ == '__main__':
    main()