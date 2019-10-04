import numpy as np 
import os

def clear_stars_files(folder_path):
    del_exts = ['.dat', '.mrf', '.irf', '.out', '.sr3']
    test = os.listdir(folder_path)
    for item in test:
        if any(item.endswith(ext) for ext in del_exts):
            try:
                os.remove(os.path.join(folder_path, item))
            except:
                pass


def copy_up_to_string(fid, fileID, i, flag_string):
    '''
    Copies contents of fid up to where it sees 'flag_string'
    '''
    flag = True
    while flag:
        print(fid[i], file = fileID, end="")
        i+=1
        if not fid[i-1].split():
            flag = True
        else:
            if fid[i-1].split()[0] == flag_string:
                flag = False
            else:
                flag = True
    return i