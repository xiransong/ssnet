import time
import numpy as np
import os
import os.path as osp

from Utils import io


class Timer:

    def __init__(self):
        self.time_start = {}
        self.time_list_dict = {}

    def start(self, event_name):
        self.time_start[event_name] = time.time()

    def end(self, event_name, verbose=False):
        event_time = time.time() - self.time_start[event_name]

        if verbose:
            print("[time][{}]: {:.1f} s".format(event_name, event_time))
        
        if event_name not in self.time_list_dict:
            self.time_list_dict[event_name] = [event_time]
        else:
            self.time_list_dict[event_name].append(event_time)

    def get_all_mean_time(self):
        mean_time_dict = {}
        for event_name in self.time_list_dict:
            mean_time_dict[event_name] = np.mean(self.time_list_dict[event_name])
        return mean_time_dict

    def save_record(self, root, prefix=""):
        if not osp.exists(root):
            os.mkdir(root)
        mean_time_dict = self.get_all_mean_time()
        io.save_json(osp.join(root, prefix + "mean_time.json"), mean_time_dict)
