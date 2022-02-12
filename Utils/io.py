import pickle as pkl
import json
import yaml
import gc
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime


def save_pkl_big_np_array(v, filename, batch_size=100000):
    print('saving {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    sizes = np.shape(v)
    dtype = v.dtype
    N = sizes[0]
    batch_num = (N-1)//batch_size + 1
    
    f = open(filename, 'wb')
    p = pkl.Pickler(f) ##-- do we need to use protocal=3?
    p.fast = True
    p.dump(sizes)
    p.dump(dtype)
    p.dump((batch_num, batch_size))
    for i in tqdm(range(batch_num), desc='batch saving'):
        start = i * batch_size
        end = min((i+1)*batch_size, N)
        p.dump(v[start:end])
    
    f.close()


def load_pkl_big_np_array(filename):
    print('loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, "rb") as f: 
        sizes = pkl.load(f)
        dtype = pkl.load(f)
        obj = np.zeros(shape=sizes, dtype=dtype)
        batch_num, batch_size = pkl.load(f)
        for i in tqdm(range(batch_num), desc='batch loading'):
            start = i * batch_size
            end = min(sizes[0], (i+1)*batch_size)
            obj[start:end] = pkl.load(f)
    
    print('finish loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    return obj


def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        gc.disable()
        obj = pkl.load(f)
        gc.enable()
    return obj


def save_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def load_yaml(filename):
    with open(filename, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(filename, obj):
    with open(filename, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)
