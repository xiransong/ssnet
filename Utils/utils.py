import torch
import os.path as osp
import pathlib
import shutil
import numpy as np
import numba


def wc_count(file_name):
    assert osp.exists(file_name)
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


def combine_dict_list_and_calc_mean(dict_list):
    d = {}
    for key in dict_list[0]:
        d[key] = np.array([
            dict_list[i][key] for i in range(len(dict_list))]).mean()
    return d


def element_wise_map(fn, t, dtype=None):
    
    if dtype is None:
        dtype = t.dtype
    new_t = torch.empty(size=t.size(), dtype=dtype)

    _t = t.view(-1).cpu().numpy()
    _new_t = new_t.view(-1).cpu().numpy()

    for i in range(len(_t)):
        _new_t[i] = fn(_t[i])

    return new_t


def assert_path_exists(path):
    
    assert osp.exists(path), "no such file or directory: {}".format(path)


def ensure_dir(path):
    
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def ensure_results_dir(results_root, config_file=None):
    ensure_dir(results_root)
    
    if config_file is not None:
        shutil.copyfile(config_file,
                        osp.join(results_root, osp.basename(config_file)))

def print_dict(d):
    for key in d:
        print(key, ":", d[key])

@numba.jit(nopython=True)
def find_first(item, vec):
    '''return the index of the first occurence of item in vec'''
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

@numba.jit(nopython=True)
def mod_with_zero_exception(a, n, padding):
    r = np.empty_like(a)
    for i in range(r.shape[0]):
        if n[i] != 0:
            r[i] = a[i] % n[i]
        else:
            r[i] = padding
    return r


class ReIndexDict:
    
    def __init__(self):
        self.cnt = 0
        self.dic = {}
    
    def __getitem__(self, old_id):
        if old_id in self.dic:
            return self.dic[old_id]
        else:
            new_id = self.cnt
            self.dic[old_id] = new_id
            self.cnt += 1
            return new_id
    
    def __len__(self):
        return len(self.dic)


def get_formatted_results(r):
    s = ""
    for key in r.keys():
        s += "{}:{:.4f} || ".format(key, r[key])
    return s
