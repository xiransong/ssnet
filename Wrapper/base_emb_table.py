from Utils import io

import torch


class EmbTable:
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        '''
        requirements:
        
        data:
            train_graph
        
        config:
            base_emb_table_device: 
            from_pretrained: (optional)
            file_pretrained_embs: (optional)
            freeze_nodes_emb: (optional)
            emb_dim: 
            embs_lr: 
            
        '''
        if 'train_graph' in data:
            self.num_nodes = data['train_graph'].num_nodes()
        else:
            self.num_nodes = data['num_nodes']
        self.device = config['base_emb_table_device']
        
        
        # build nodes_embs
        self.emb_dim = config['emb_dim']
        use_sparse_emb = bool('use_sparse_emb' in config and config['use_sparse_emb'])
        from_pretrained = bool('from_pretrained' in config and config['from_pretrained'])
        freeze_nodes_emb = bool('freeze_nodes_emb' in config and config['freeze_nodes_emb'])
        
        if from_pretrained:
            pretrained_embs = torch.load(config['file_pretrained_embs'], map_location='cpu')
            nodes_embs = torch.nn.Embedding.from_pretrained(
                pretrained_embs, freeze=freeze_nodes_emb, sparse=use_sparse_emb)
        else:
            nodes_embs = torch.nn.Embedding(self.num_nodes, self.emb_dim, sparse=use_sparse_emb)  # initialized from N(0, 1)
        self.nodes_embs = nodes_embs.to(self.device)
        
        self.opt_param_list = {'dense':[], 'sparse':[]}
        if not freeze_nodes_emb:
            _p = {'params': self.nodes_embs.weight, 'lr': config['embs_lr']}
            if use_sparse_emb:
                self.opt_param_list['sparse'].append(_p)
            else:
                self.opt_param_list['dense'].append(_p)
        
        self.shape = self.size()
    
    def parameters(self):
        return self.opt_param_list
    
    def __getitem__(self, idx):
        return self.nodes_embs(idx.to(self.device))
    
    def size(self):
        return torch.Size([self.num_nodes, self.emb_dim])
    
    def __len__(self):
        return self.shape[0]
    
    def get_full_matrix(self):
        return self.nodes_embs.weight

    def del_grad_matrix(self):
        del self.nodes_embs.weight.grad
