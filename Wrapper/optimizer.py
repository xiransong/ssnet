import torch


class DoNothingOptWrapper:
    
    def __init__(self):
        pass
    
    def build(self, opt_param_list, data, config):
        pass
    
    def zero_grad(self):
        pass
    
    def step(self):
        pass


class AdamWrapper:
    
    def __init__(self):
        pass

    def build(self, opt_param_list, data, config):
        if 'clip' in config:
            self.max_norm_clip = config['clip']
        else:
            self.max_norm_clip = None
        
        print("## using Adam optimizer")
        _params = opt_param_list
        if isinstance(_params, list):
            if 'use_sparse_emb' in config and config['use_sparse_emb']:
                self.opt_list = [torch.optim.SparseAdam(_params)]
            else:
                self.opt_list = [torch.optim.Adam(_params)]
        else:
            self.opt_list = []
            if len(_params['dense']):
                self.opt_list.append(torch.optim.Adam(_params['dense']))
            if len(_params['sparse']):
                self.opt_list.append(torch.optim.SparseAdam(_params['sparse']))
        
    def zero_grad(self):
        for opt in self.opt_list:
            opt.zero_grad()
        
    def step(self):
        for opt in self.opt_list:
            if self.max_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],
                                               max_norm=self.max_norm_clip, norm_type=2.0)
            opt.step()


class SGDWrapper:
    
    def __init__(self):
        pass

    def build(self, opt_param_list, data, config):
        if 'clip' in config:
            self.max_norm_clip = config['clip']
        else:
            self.max_norm_clip = None
        
        print("## using SGD optimizer")
        if isinstance(opt_param_list, list):
            self.opt = torch.optim.SGD(opt_param_list)
        else:
            self.opt = torch.optim.SGD(
                opt_param_list['dense'] + opt_param_list['sparse']
            )

    def zero_grad(self):
        self.opt.zero_grad()
        
    def step(self):
        if self.max_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]['params'], 
                                           max_norm=self.max_norm_clip, norm_type=2.0)
        self.opt.step()
